import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# --- Constants ---
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Functions ---

@st.cache_resource
def load_embedding_model():
    """Loads the SentenceTransformer model."""
    with st.spinner("正在載入AI模型... (Loading AI model...)"):
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

@st.cache_resource
def create_faiss_index(_embeddings):
    """Creates a FAISS index from embeddings."""
    d = _embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(_embeddings)
    return index

def load_documents(folder_path):
    """Reads .doc and .docx files."""
    doc_texts, doc_names = [], []
    files = [f for f in os.listdir(folder_path) if (f.endswith('.docx') or f.endswith('.doc')) and not f.startswith('~$')]
    for filename in files:
        try:
            full_path = os.path.join(folder_path, filename)
            doc = docx.Document(full_path)
            full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            doc_texts.append(full_text)
            doc_names.append(filename)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
    return doc_names, doc_texts

def split_text(doc_names, doc_texts):
    """Splits documents into text chunks."""
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def search_index(query, model, index, chunks, k=5):
    """Searches the FAISS index."""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    # Collect unique chunks
    unique_indices = list(dict.fromkeys(indices[0]))
    
    results = [chunks[i] for i in unique_indices]
    return results

def generate_answer(query, context):
    """Generates an answer using the Gemini API based on provided context."""
    prompt = f"""
    You are a professional assistant for answering questions about Electrical Discharge Machining (EDM) operations.
    Please answer the user's question based *only* on the following context retrieved from the operation manuals.
    If the context does not contain the answer, state that clearly and do not add any information.
    Be concise and helpful.

    --- CONTEXT ---
    {context}
    --- END CONTEXT ---

    Here is the user's question:
    --- QUESTION ---
    {query}
    --- END QUESTION ---

    Your answer (in Traditional Chinese):
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"產生答案時發生錯誤：{e}"

# --- Streamlit App ---

st.title("放電機操作手冊問答小幫手")
st.header("EDM Manual Q&A Assistant")

# --- Sidebar for API Key ---
st.sidebar.header("AI模型設定 (AI Model Settings)")
google_api_key = st.sidebar.text_input("輸入您的 Google API Key:", type="password")
st.sidebar.markdown("""
[如何取得您的Google API Key](https://aistudio.google.com/app/apikey)
""")

# --- Initialization and Data Loading ---
def initialize(api_key):
    genai.configure(api_key=api_key)
    st.session_state.model = load_embedding_model()

    docs_folder = os.path.dirname(os.path.abspath(__file__))
    doc_names, doc_texts = load_documents(docs_folder)
    
    if not doc_texts:
        st.warning("資料夾中找不到任何 .doc 或 .docx 文件。")
        st.stop()
        
    with st.spinner("正在分析文件並建立AI索引... (This may take a moment)"):
        chunks, chunk_sources = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks, show_progress_bar=True)
        
        st.session_state.chunks = chunks
        st.session_state.chunk_sources = chunk_sources
        st.session_state.embeddings = np.array(embeddings)
        st.session_state.faiss_index = create_faiss_index(st.session_state.embeddings)
    
    st.session_state.initialized = True
    st.success(f"AI助理準備就緒！已分析 {len(chunks)} 個文本段落。")

if google_api_key:
    if 'initialized' not in st.session_state:
        initialize(google_api_key)
else:
    st.info("請在左側側邊欄輸入您的 Google API Key 以啟動AI助理。")
    st.stop()

st.divider()

# --- Q&A Interface ---
st.header("請在此提問 (Ask a Question)")

query = st.text_input("輸入您的問題 (Your question):", "")

if st.button("查詢 (Search)"):
    if query and st.session_state.get('initialized', False):
        with st.spinner("正在搜尋答案... (Searching for answer...)"):
            # 1. Retrieve context
            retrieved_chunks = search_index(
                query,
                st.session_state.model,
                st.session_state.faiss_index,
                st.session_state.chunks,
                k=5 # Retrieve more chunks for better context
            )
            context_for_prompt = "\n\n".join(retrieved_chunks)

            # 2. Generate answer
            answer = generate_answer(query, context_for_prompt)
            
            # 3. Display results
            st.subheader("AI助理的答案 (AI Assistant's Answer):")
            st.success(answer)
            
            with st.expander("查看AI參考的資訊來源 (View sources used by AI)"):
                st.write(retrieved_chunks)
            
    elif not query:
        st.warning("請輸入問題。(Please enter a question.)")

