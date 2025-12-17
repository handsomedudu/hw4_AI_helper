import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# ==========================================
# 1. 這裡直接寫死您的 API KEY
# ==========================================
MY_API_KEY = "AIzaSyBVF_HR40eAuH_MmevkgWe5E33Ielm0eCw" 

# 設定 Google Gemini
genai.configure(api_key=MY_API_KEY)

# 向量模型名稱
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- 核心功能函式 ---

@st.cache_resource
def load_embedding_model():
    """載入語意分析模型"""
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def create_faiss_index(_embeddings):
    """建立搜尋索引"""
    d = _embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(_embeddings)
    return index

def load_documents(folder_path):
    """讀取當前目錄下的 Word 文件"""
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
            st.error(f"讀取 {filename} 出錯: {e}")
    return doc_names, doc_texts

def split_text(doc_names, doc_texts):
    """將長文本切割成段落"""
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def generate_answer(query, context):
    """調用 Gemini 生成答案"""
    prompt = f"""
    你是一位專業的放電機 (EDM) 操作助手。
    請僅根據以下手冊內容回答問題。如果手冊沒提到，請說不知道。
    
    --- 手冊內容 ---
    {context}
    ---
    問題：{query}
    回答（繁體中文）：
    """
    # 這裡直接使用 1.5-flash，確保不會有 404 錯誤
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)
    return response.text

# --- 初始化流程 ---
if 'initialized' not in st.session_state:
    with st.spinner("系統初始化中，請稍候..."):
        # 1. 載入模型
        st.session_state.model = load_embedding_model()
        # 2. 讀取文件 (app.py 所在目錄)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("錯誤：找不到 .docx 手冊檔案！")
            st.stop()
            
        # 3. 建立向量資料庫
        chunks, _ = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        st.session_state.initialized = True

# --- 使用者介面 ---
st.set_page_config(page_title="放電機 AI 助理")
st.title("⚡ 放電機操作 AI 小幫手")
st.info("本系統已內建 AI 授權，直接輸入問題即可。")

query = st.text_input("請輸入您的操作問題：", placeholder="例如：如何進行尋邊操作？")

if st.button("詢問 AI"):
    if query:
        with st.spinner("搜尋手冊內容並分析中..."):
            # 檢索最相關的 5 個片段
            query_embedding = st.session_state.model.encode([query])
            distances, indices = st.session_state.faiss_index.search(query_embedding, 5)
            
            context = "\n\n".join([st.session_state.chunks[i] for i in indices[0] if i != -1])
            
            # 生成答案
            answer = generate_answer(query, context)
            
            st.markdown("### 🤖 回答結果：")
            st.success(answer)
            
            with st.expander("查看參考來源段落"):
                st.write(context)
    else:
        st.warning("請先輸入問題。")