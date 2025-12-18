import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# ==========================================
# 1. API KEY 寫死設定
# ==========================================
MY_API_KEY = "AIzaSyBVF_HR40eAuH_MmevkgWe5E33Ielm0eCw" 
genai.configure(api_key=MY_API_KEY)

# 向量模型名稱
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- 核心功能函式 ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

@st.cache_resource
def create_faiss_index(_embeddings):
    d = _embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(_embeddings)
    return index

def load_documents(folder_path):
    """只讀取 .docx 檔案，避免 .doc 導致的錯誤"""
    doc_texts, doc_names = [], []
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
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def get_best_model_name():
    """最終偵錯步驟：動態尋找可用的 Gemini 模型名稱"""
    try:
        for m in genai.list_models():
            # 優先尋找 1.5-flash，若無則找 1.5-pro
            if 'gemini-1.5-flash' in m.name.lower():
                return m.name
        return "gemini-1.5-flash" # 保底
    except:
        return "gemini-1.5-flash"

def generate_answer(query, context):
    """調用偵測到的正確模型名稱"""
    target_model = get_best_model_name()
    prompt = f"你是一位放電機助手。請根據手冊回答問題：\n\n{context}\n\n問題：{query}\n回答（繁體中文）："
    
    try:
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"產生答案時發生錯誤：{e}\n(嘗試使用的模型名稱為: {target_model})"

# --- 初始化 ---
st.set_page_config(page_title="放電機 AI 助理", layout="centered")
st.title("⚡ 放電機操作 AI 小幫手 (Debug 版)")

if 'initialized' not in st.session_state:
    with st.spinner("系統初始化中..."):
        st.session_state.model = load_embedding_model()
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("找不到 .docx 檔案！請確認已將 .doc 另存為 .docx 並上傳。")
            st.stop()
            
        chunks, _ = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        st.session_state.initialized = True
        st.success(f"✅ 已載入 {len(doc_names)} 份手冊。")

# --- UI ---
query = st.text_input("請輸入您的問題：")
if st.button("詢問"):
    if query:
        with st.spinner("分析中..."):
            query_embedding = st.session_state.model.encode([query])
            _, indices = st.session_state.faiss_index.search(query_embedding, 5)
            context = "\n\n".join([st.session_state.chunks[i] for i in indices[0]])
            answer = generate_answer(query, context)
            st.markdown("### 🤖 回答：")
            st.info(answer)

# --- 最終偵錯資訊 (Debug Info) ---
with st.expander("🛠️ 系統診斷資訊 (最終偵錯步驟)"):
    st.write(f"當前使用的模型名稱: `{get_best_model_name()}`")
    st.write(f"已載入的段落數量: {len(st.session_state.chunks)}")
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        st.write("您的 API Key 可用的模型清單：")
        st.json(models)
    except Exception as e:
        st.write(f"無法獲取模型清單: {e}")