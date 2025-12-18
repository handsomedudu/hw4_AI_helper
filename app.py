import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# --- 1. 從 Streamlit Secrets 安全讀取 API Key ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 錯誤：未在 Streamlit Cloud 的 Secrets 中找到 GOOGLE_API_KEY。請先完成設定。")
    st.stop()

# 向量模型名稱 (適合繁體中文手冊)
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- 核心功能函式 ---

@st.cache_resource
def load_embedding_model():
    """載入語意分析模型"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def create_faiss_index(_embeddings):
    """建立 FAISS 高速檢索索引"""
    d = _embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(_embeddings)
    return index

def load_documents(folder_path):
    """讀取當前目錄下的 .docx 手冊檔案"""
    doc_texts, doc_names = [], []
    # 僅篩選 .docx 檔案，避免 python-docx 無法讀取舊版 .doc 導致崩潰
    files = [f for f in os.listdir(folder_path) if f.endswith('.docx') and not f.startswith('~$')]
    
    for filename in files:
        try:
            full_path = os.path.join(folder_path, filename)
            doc = docx.Document(full_path)
            full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            doc_texts.append(full_text)
            doc_names.append(filename)
        except Exception as e:
            st.error(f"讀取 {filename} 出錯: {e}")
            
    return doc_names, doc_texts

def split_text(doc_names, doc_texts):
    """將長文本切割成段落 (Chunks)"""
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def generate_answer(query, context):
    """調用 Gemini 2.0 Flash 生成答案"""
    # 根據偵測清單，models/gemini-2.0-flash 是您帳號目前最穩定且高效的選擇
    target_model = "models/gemini-2.0-flash"
    
    prompt = f"""
    你是一位專業的放電機 (EDM) 操作助手。
    請僅根據以下手冊內容回答使用者的問題。如果手冊內容中沒有答案，請禮貌地告知你不知道。
    
    --- 手冊內容 (CONTEXT) ---
    {context}
    ---
    
    問題：{query}

    回答（請使用繁體中文）：
    """
    try:
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 產生答案時發生錯誤：{e}"

# --- 初始化流程 ---

st.set_page_config(page_title="放電機 AI 助理", page_icon="⚡")
st.title("⚡ 放電機操作 AI 小幫手")
st.caption("使用 RAG 技術與 Gemini 2.0 Flash 模型，專注於提供安全、精準的手冊解答。")

if 'initialized' not in st.session_state:
    with st.spinner("系統正在分析文件並建立索引，請稍候..."):
        # 1. 載入語意模型
        st.session_state.model = load_embedding_model()
        
        # 2. 讀取 .docx 檔案
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("⚠️ 找不到 .docx 檔案！請確保已將 .doc 另存新檔為 .docx 並上傳至 GitHub。")
            st.stop()
            
        # 3. 建立向量資料庫
        chunks, _ = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks, show_progress_bar=False)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        
        st.session_state.initialized = True
        st.success(f"✅ 初始化完成！已載入 {len(doc_names)} 份手冊。")

# --- 使用者介面 ---

st.divider()
query = st.text_input("請輸入您的操作問題：", placeholder="例如：如何進行工件尋邊？")

if st.button("詢問 AI"):
    if query:
        with st.spinner("正在搜尋答案..."):
            # 向量檢索
            query_embedding = st.session_state.model.encode([query])
            distances, indices = st.session_state.faiss_index.search(query_embedding, 5)
            
            # 獲取最相關的段落
            retrieved_chunks = [st.session_state.chunks[i] for i in indices[0] if i != -1]
            context = "\n\n".join(retrieved_chunks)
            
            # 生成答案
            answer = generate_answer(query, context)
            
            st.markdown("### 🤖 AI 的回答：")
            st.success(answer)
            
            with st.expander("🔍 查看參考來源段落"):
                for idx, chunk in enumerate(retrieved_chunks):
                    st.info(f"來源片段 {idx+1}:\n{chunk}")
    else:
        st.warning("請先輸入問題內容。")