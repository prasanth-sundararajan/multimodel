import os
import io
import re
import shutil
import base64
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
import pymupdf  # Replaces deprecated 'import fitz'
from dotenv import load_dotenv

import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------------------------------------------------
# 0. CONFIGURATION & TRACING
# -----------------------------------------------------------------------------
load_dotenv()

PERSIST_DIRECTORY = "./chroma_agri_db"
COLLECTION_NAME = "tn_agri_schemes"
TNAU_SEED_URL = "https://agritech.tnau.ac.in/expert_system/paddy/Schemes.html"

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "tn-agri-schemes-rag")

# -----------------------------------------------------------------------------
# STEP 1: SCHEME EXTRACTION & SANITIZATION HELPERS
# -----------------------------------------------------------------------------
SCHEME_KEYWORDS = [
    "scheme", "subsidy", "yojana", "mission", "assistance", "kisan",
    "component", "insurance", "grant", "திட்டம்", "மானியம்",
    "development programme", "development program", "procurement",
    "mechanisation", "mechanization", "nadp", "nais", "rkvy", "smam",
    "pmfby", "nfsm", "pmksy"
]

EXCLUDE_WORDS = [
    "sl no", "name of the scheme", "scheme details", "how to apply",
    "botany", "fermented", "idli", "season and varieties", "losses from rice",
    "weed management", "nutrient management", "table 1", "table 2"
]

def clean_scheme_name(title: str) -> str:
    """Removes table header prefixes, serial numbers, and linebreaks."""
    if not title:
        return ""
    t = title.strip()
    t = re.sub(r"^(Scheme Table:\s*|\d+[\.\)]\s*)+", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"^(Name of Scheme\s*:\s*|Details of\s+)", "", t, flags=re.IGNORECASE).strip()
    t = re.sub(r"\s+", " ", t.split("\n")[0].split(";")[0]).strip()
    return t

def is_valid_scheme_button(title: str) -> bool:
    """Only display actual scheme titles as clickable sidebar buttons."""
    if not title or len(title) < 6:
        return False
    lower = title.lower()
    if any(ex in lower for ex in EXCLUDE_WORDS):
        return False
    return any(k in lower for k in SCHEME_KEYWORDS)

# -----------------------------------------------------------------------------
# STEP 2: MULTIMODAL EXTRACTION & CHILD-URL CRAWLER
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Removes irregular whitespace, non-breaking spaces, and blank lines."""
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def describe_image_with_vision(image_bytes: bytes, context_source: str, page_num: int) -> str:
    """Uses GPT-4o-mini Vision to generate descriptions of agricultural scheme diagrams."""
    try:
        b64_img = base64.b64encode(image_bytes).decode("utf-8")
        vision_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
        
        prompt = (
            "You are an agricultural scheme expert. Describe this image extracted from a "
            "Tamil Nadu government agricultural scheme PDF. Detail any machinery, tools, "
            "subsidy charts, process flow, or crop conditions shown."
        )
        msg = vision_llm.invoke([
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}
        ])
        return msg.content
    except Exception as e:
        return f"[Image description omitted: {e}]"


def extract_media_and_text_from_pdf(pdf_bytes: bytes, file_name: str):
    """Extracts text, hyperlinks, URLs, images, and audio/video references from PDFs."""
    documents = []
    discovered_urls = set()

    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    for page_idx, page in enumerate(doc):
        page_num = page_idx + 1
        page_text = clean_text(page.get_text() or "")

        for link in page.get_links():
            uri = link.get("uri")
            if uri and (uri.startswith("http://") or uri.startswith("https://")):
                discovered_urls.add(uri)

        for u in re.findall(r"https?://[^\s<>\"']+", page_text):
            discovered_urls.add(u.rstrip(".,;)"))

        if page_text:
            documents.append(
                Document(
                    page_content=page_text,
                    metadata={"source": file_name, "page": page_num, "type": "pdf_text", "scheme_title": file_name}
                )
            )

        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            img_desc = describe_image_with_vision(image_bytes, file_name, page_num)
            documents.append(
                Document(
                    page_content=f"Image Description (Page {page_num}, Img {img_idx+1}): {img_desc}",
                    metadata={"source": file_name, "page": page_num, "type": "pdf_image_caption", "scheme_title": file_name}
                )
            )

    doc.close()

    for url in list(discovered_urls):
        url_lower = url.lower()
        if any(ext in url_lower for ext in [".mp4", ".mov", ".avi", "youtube.com", "youtu.be"]):
            documents.append(
                Document(
                    page_content=f"Instructional Farm Video Link: {url}.",
                    metadata={"source": file_name, "type": "video_reference", "url": url, "scheme_title": file_name}
                )
            )
        elif any(ext in url_lower for ext in [".mp3", ".wav"]):
            documents.append(
                Document(
                    page_content=f"Agricultural Radio/Audio Broadcast Link: {url}.",
                    metadata={"source": file_name, "type": "audio_reference", "url": url, "scheme_title": file_name}
                )
            )

    return documents, list(discovered_urls)


def parse_schemes_from_html(soup: BeautifulSoup, page_url: str):
    """Parses schemes from HTML without dropping contextual body data."""
    scheme_docs = []
    page_title = soup.title.string.strip() if soup.title and soup.title.string else page_url
    cleaned_page_title = clean_scheme_name(page_title)

    # Strategy A: Capture tables
    tables = soup.find_all("table")
    for tbl_idx, table in enumerate(tables):
        rows = table.find_all("tr")
        if len(rows) > 1:
            table_text = clean_text(table.get_text())
            header_text = clean_text(rows[0].get_text())[:100]
            tbl_title = clean_scheme_name(header_text) or cleaned_page_title

            scheme_docs.append(
                Document(
                    page_content=f"Scheme Page: {page_title}\nSection: {header_text}\n\nFull Details:\n{table_text}",
                    metadata={
                        "source": page_url,
                        "scheme_title": tbl_title,
                        "type": "scheme_table"
                    }
                )
            )

    # Strategy B: Capture Headings & Content Paragraphs
    headings = soup.find_all(["h2", "h3", "h4", "b", "strong"])
    for h in headings:
        h_text = clean_text(h.get_text())
        cleaned_h = clean_scheme_name(h_text)
        if len(cleaned_h) > 6:
            content_parts = []
            for sib in h.next_siblings:
                if getattr(sib, 'name', None) in ["h2", "h3", "h4", "b", "strong"]:
                    break
                if hasattr(sib, 'get_text'):
                    content_parts.append(sib.get_text())
            
            body = clean_text(" ".join(content_parts))
            if len(body) > 40:
                scheme_docs.append(
                    Document(
                        page_content=f"Scheme Name: {cleaned_h}\nDetails, Subsidies & Eligibility:\n{body}",
                        metadata={
                            "source": page_url,
                            "scheme_title": cleaned_h,
                            "type": "scheme_section"
                        }
                    )
                )

    # Strategy C: Full page fallback (ensures nothing is lost)
    for tag in soup(["script", "style", "nav", "footer", "aside"]):
        tag.decompose()
    page_body = clean_text(soup.get_text())
    if len(page_body) > 80:
        scheme_docs.append(
            Document(
                page_content=f"Scheme Documentation: {page_title}\n\nFull Content:\n{page_body}",
                metadata={"source": page_url, "scheme_title": cleaned_page_title, "type": "web_page"}
            )
        )

    return scheme_docs


def deep_crawl_tnau_portal(seed_url=TNAU_SEED_URL, max_pages=35):
    """Crawls TNAU schemes, unrolling frames and nested child URLs."""
    session = requests.Session()
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    discovered_docs = []
    visited_urls = set()
    queue = [seed_url]

    while queue and len(visited_urls) < max_pages:
        curr_url = queue.pop(0)
        if curr_url in visited_urls:
            continue
        visited_urls.add(curr_url)

        try:
            resp = session.get(curr_url, headers=headers, timeout=12)
            resp.encoding = resp.apparent_encoding or "utf-8"
            if resp.status_code != 200:
                continue

            # Follow PDF child files directly
            if curr_url.lower().endswith(".pdf") or "application/pdf" in resp.headers.get("Content-Type", ""):
                file_title = curr_url.split("/")[-1]
                pdf_docs, inner_links = extract_media_and_text_from_pdf(resp.content, file_title)
                discovered_docs.extend(pdf_docs)
                for l in inner_links:
                    if l not in visited_urls and l.startswith("http"):
                        queue.append(l)
                continue

            soup = BeautifulSoup(resp.text, "html.parser")

            # Follow frames & iframes
            for frame in soup.find_all(["frame", "iframe"]):
                src = frame.get("src")
                if src:
                    f_url = urljoin(curr_url, src)
                    if f_url not in visited_urls and f_url not in queue:
                        queue.append(f_url)

            # Follow child links within TNAU and TN gov domains
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                if href.startswith(("#", "javascript:", "mailto:")):
                    continue
                full_url = urljoin(curr_url, href)
                if any(dom in full_url for dom in ["agritech.tnau.ac.in", "tn.gov.in"]):
                    if full_url not in visited_urls and full_url not in queue:
                        queue.append(full_url)

            page_schemes = parse_schemes_from_html(soup, curr_url)
            discovered_docs.extend(page_schemes)

        except Exception as e:
            print(f"Error crawling {curr_url}: {e}")

    return discovered_docs

# -----------------------------------------------------------------------------
# STEP 3: PERSISTENT VECTOR STORE INGESTION
# -----------------------------------------------------------------------------
def get_vector_store():
    """Returns persistent ChromaDB instance."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )


def ingest_documents(documents):
    """Splits documents and commits to Chroma vector store."""
    if not documents:
        return 0

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1400,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    vector_store = get_vector_store()
    vector_store.add_documents(split_docs)
    return len(split_docs)

# -----------------------------------------------------------------------------
# STEP 4: THOROUGH RAG PIPELINE
# -----------------------------------------------------------------------------
RAG_SYSTEM_PROMPT = """You are an expert Tamil Nadu Agriculture Schemes Consultant.

Answer the user's inquiry thoroughly and comprehensively using the retrieved context below.

Instructions:
1. Provide a detailed, complete breakdown:
   - Official Scheme Name & Objective
   - Subsidies & Financial Assistance (percentages, amounts, equipment covered)
   - Eligibility Criteria (small/marginal farmers, women, SC/ST, landholding limits)
   - Documents Required & How to Apply
   - Officers / Departments to be approached (e.g., ADA, AAO, Block Agricultural Office)
2. Always synthesize and summarize all relevant points present in the context. Never give a one-line rejection if related information is available.
3. Only state that you have insufficient information if the topic is completely absent from the context.
4. Support both Tamil and English natively depending on the question's language.
5. If visual diagram details or media links (videos/audio) exist in the context, cite them clearly.

Retrieved Context:
{context}

User Inquiry:
{question}

Comprehensive Answer:"""


def format_docs(docs):
    formatted = []
    for d in docs:
        src = d.metadata.get("source", "Unknown")
        sch_title = d.metadata.get("scheme_title", "")
        header = f"--- Source: {src}" + (f" | Scheme: {sch_title}" if sch_title else "") + " ---"
        formatted.append(f"{header}\n{d.page_content}")
    return "\n\n".join(formatted)


def build_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )
    prompt = ChatPromptTemplate.from_template(RAG_SYSTEM_PROMPT)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever

# -----------------------------------------------------------------------------
# STEP 5: STREAMLIT UI IMPLEMENTATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Tamil Nadu Agriculture Schemes Assistant",
    page_icon="🌾",
    layout="wide"
)

st.markdown("""
<style>
div[data-testid="stSidebarUserContent"] button[kind="secondary"] {
    text-align: left;
    white-space: normal;
    height: auto;
    padding-top: 6px;
    padding-bottom: 6px;
    margin-bottom: 4px;
    font-size: 0.85rem;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "portal_auto_ingested" not in st.session_state:
    st.session_state.portal_auto_ingested = False


def run_full_reingest():
    """Wipes old index and crawls TNAU schemes fresh."""
    if os.path.exists(PERSIST_DIRECTORY):
        try:
            shutil.rmtree(PERSIST_DIRECTORY)
        except Exception:
            pass

    with st.status("Rebuilding vector database and crawling TNAU schemes...", expanded=True) as status:
        st.write(f"Initiating deep crawl from: `{TNAU_SEED_URL}`")
        crawled_docs = deep_crawl_tnau_portal(seed_url=TNAU_SEED_URL, max_pages=35)
        st.write(f"Collected {len(crawled_docs)} scheme sections and child URLs. Indexing...")
        chunks_count = ingest_documents(crawled_docs)
        status.update(label=f"Done! Ingested {chunks_count} complete scheme chunks.", state="complete")

    st.session_state.portal_auto_ingested = True
    st.success("TNAU Portal indexing completed!")
    st.rerun()


st.title("🌾 Tamil Nadu Agriculture Schemes Assistant")
st.caption("AI Assistant answering farmer questions strictly from indexed TNAU schemes, child URLs, and multimodal PDFs.")

# Sidebar Controls
with st.sidebar:
    st.header("⚙️ Data Ingestion Pipeline")
    
    st.subheader("1. Crawl TNAU Schemes Portal")
    try:
        vs_check = get_vector_store()
        initial_chunk_count = vs_check._collection.count()
    except Exception:
        initial_chunk_count = 0

    if initial_chunk_count == 0 and not st.session_state.portal_auto_ingested:
        run_full_reingest()

    if st.button("🔄 Clean Re-Index TNAU Portal", type="primary", use_container_width=True):
        run_full_reingest()

    st.markdown("---")

    st.subheader("2. Upload Scheme PDFs")
    uploaded_files = st.file_uploader(
        "Upload Scheme PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Index Uploaded PDFs & Multimodal Media", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        else:
            with st.status("Extracting PDF text, child URLs, and images via PyMuPDF...", expanded=True) as status:
                all_pdf_docs = []
                for f in uploaded_files:
                    st.write(f"Reading: `{f.name}`")
                    bytes_data = f.read()
                    docs, child_urls = extract_media_and_text_from_pdf(bytes_data, f.name)
                    all_pdf_docs.extend(docs)

                    for link in child_urls[:5]:
                        st.write(f"Crawling child link inside PDF: `{link[:50]}...`")
                        session = requests.Session()
                        try:
                            res = session.get(link, timeout=10)
                            if res.status_code == 200:
                                soup = BeautifulSoup(res.text, "html.parser")
                                child_docs = parse_schemes_from_html(soup, link)
                                all_pdf_docs.extend(child_docs)
                        except Exception as e:
                            st.write(f"Could not reach {link}: {e}")

                chunks = ingest_documents(all_pdf_docs)
                status.update(label=f"Completed! Indexed {chunks} chunks.", state="complete")
            st.success("PDFs indexed successfully!")
            st.rerun()

    st.markdown("---")

    st.subheader("📊 Indexed Schemes Inspector")
    try:
        vs = get_vector_store()
        col = vs._collection
        count = col.count()
        st.metric(label="Total Chunks in DB", value=count)

        if count > 0:
            with st.expander("🏛️ View & Query Verified Schemes", expanded=True):
                st.caption("Click any verified scheme to view comprehensive subsidies & eligibility:")
                all_metas = col.get(include=["metadatas"])["metadatas"]
                
                unique_schemes = set()
                for m in all_metas:
                    raw_title = m.get("scheme_title") or ""
                    cleaned = clean_scheme_name(raw_title)
                    if cleaned and is_valid_scheme_button(cleaned):
                        unique_schemes.add(cleaned)

                sorted_schemes = sorted(list(unique_schemes))

                if sorted_schemes:
                    for idx, sch in enumerate(sorted_schemes):
                        btn_label = (sch[:45] + "...") if len(sch) > 48 else sch
                        if st.button(f"🌾 {idx+1}. {btn_label}", key=f"sch_btn_{idx}", use_container_width=True):
                            # Comprehensive inquiry prompt that yields deep answers
                            st.session_state.pending_query = (
                                f"Explain the scheme '{sch}' in complete detail. "
                                f"What are its subsidies, eligibility rules, benefits, and officers to approach?"
                            )
                            st.rerun()
                else:
                    st.info("No standalone scheme headers parsed yet. Click 'Clean Re-Index TNAU Portal' above.")
        else:
            st.info("Vector DB is empty. Run an ingestion above to load schemes.")
    except Exception as e:
        st.caption("Database initializing...")

    st.markdown("---")
    st.write("### 🔍 LangSmith Tracing")
    if os.getenv("LANGCHAIN_API_KEY"):
        st.success(f"Project: `{os.getenv('LANGCHAIN_PROJECT', 'tn-agri-schemes-rag')}`")
    else:
        st.caption("Add `LANGCHAIN_API_KEY` to `.env` to enable full trace observability.")

# Main Chat Display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Referenced Sources"):
                for src in msg["sources"]:
                    st.write(f"- `{src}`")

# Catch query
active_query = None
chat_input_val = st.chat_input("Enter your query (e.g., What is the subsidy for paddy transplanters? / நெல் நடவு இயந்திர மானியம் என்ன?)")

if st.session_state.pending_query:
    active_query = st.session_state.pending_query
    st.session_state.pending_query = None
elif chat_input_val:
    active_query = chat_input_val

# Process Query
if active_query:
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    vector_store = get_vector_store()
    rag_chain, retriever = build_rag_chain(vector_store)

    with st.chat_message("assistant"):
        with st.spinner("Searching indexed schemes in vector store..."):
            retrieved_docs = retriever.invoke(active_query)
            answer = rag_chain.invoke(active_query)

            st.markdown(answer)

            cited_sources = []
            for d in retrieved_docs:
                title = d.metadata.get("scheme_title")
                src = d.metadata.get("source", "Unknown")
                label = f"{title} ({src})" if title else src
                if label not in cited_sources:
                    cited_sources.append(label)

            if cited_sources:
                with st.expander("Referenced Scheme Sources"):
                    for s in cited_sources:
                        st.write(f"- `{s}`")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": cited_sources
            })