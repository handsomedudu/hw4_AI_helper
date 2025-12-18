import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# ==========================================
# 1. API KEY 寫死設定 (請確認此 Key 有效)
# ==========================================
MY_API_KEY = "AIzaSyBVF_HR40eAuH_MmevkgWe5E33Ielm0eCw" 
genai.configure(api_key=MY_API_KEY)

# 向量模型名稱 (多國語言版)
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
    # 僅篩選 .docx 檔案，避開無法讀取的舊版 .doc
    files = [f for f in os.listdir(folder_path) if f.endswith('.docx') and not f.startswith('~$')]
    
    for filename in files:
        try:
            full_path = os.path.join(folder_path, filename)
            doc = docx.Document(full_path)
            full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            doc_texts.append(full_text)
            doc_names.append(filename)
        except Exception as e:
            st.error(f"讀取 {filename} 出錯 (請確認是否為 .docx 格式): {e}")
            
    return doc_names, doc_texts

def split_text(doc_names, doc_texts):
    """將文本切割成段落 (Chunks)"""
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def get_best_model_name():
    """根據您的診斷清單，自動選取最合適的模型"""
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        # 優先順序：2.0-flash > flash-latest > 1.5-flash
        priority_list = [
            "models/gemini-2.0-flash", 
            "models/gemini-flash-latest", 
            "models/gemini-1.5-flash"
        ]
        for target in priority_list:
            if target in available_models:
                return target
        return available_models[0] if available_models else "models/gemini-2.0-flash"
    except:
        return "models/gemini-2.0-flash"

def generate_answer(query, context):
    """調用 Gemini 生成答案"""
    target_model = get_best_model_name()
    prompt = f"""
    你是一位專業的放電機 (EDM) 操作助手。
    請僅根據以下手冊內容回答問題。如果手冊內容中沒有答案，請說不知道。
    
    --- 手冊內容 (CONTEXT) ---
    {context}
    ---
    
    問題：{query}

    回答（繁體中文）：
    """
    try:
        model = genai.GenerativeModel(target_model)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"產生答案時發生錯誤：{e}\n(嘗試模型: {target_model})"

# --- 初始化流程 ---

st.set_page_config(page_title="放電機 AI 助理", page_icon="⚡")
st.title("⚡ 放電機操作 AI 小幫手")
st.caption("基於 RAG 技術與 Gemini 2.0 的專業問答系統")

if 'initialized' not in st.session_state:
    with st.spinner("系統初始化中..."):
        st.session_state.model = load_embedding_model()
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("找不到 .docx 檔案！請確認已將手冊上傳至目錄。")
            st.stop()
            
        chunks, _ = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        st.session_state.initialized = True
        st.success(f"✅ 已成功分析 {len(st.session_state.chunks)} 個段落。")

# --- UI 介面 ---

query = st.text_input("請輸入操作問題：", placeholder="例如：如何進行工件尋邊？")

if st.button("詢問 AI"):
    if query:
        with st.spinner("搜尋手冊中..."):
            query_embedding = st.session_state.model.encode([query])
            _, indices = st.session_state.faiss_index.search(query_embedding, 5)
            context = "\n\n".join([st.session_state.chunks[i] for i in indices[0] if i != -1])
            
            answer = generate_answer(query, context)
            
            st.markdown("### 🤖 AI 回答：")
            st.success(answer)
            
            with st.expander("🔍 查看參考來源段落"):
                st.write(context)
    else:
        st.warning("請輸入問題內容。")

# --- 診斷資訊 ---
with st.expander("🛠️ 系統狀態診斷"):
    st.write(f"當前自動選用模型: `{get_best_model_name()}`")
    st.write(f"手冊段落總數: {len(st.session_state.chunks)}")