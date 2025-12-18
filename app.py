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

# 設定 Google Gemini 配置
genai.configure(api_key=MY_API_KEY)

# 向量模型名稱 (多國語言版)
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- 核心功能函式 ---

@st.cache_resource
def load_embedding_model():
    """載入語意分析模型"""
    with st.spinner("正在載入AI模型..."):
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return model

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
    # 修正：僅篩選 .docx 檔案，避開無法讀取的 .doc
    files = [f for f in os.listdir(folder_path) if f.endswith('.docx') and not f.startswith('~$')]
    
    for filename in files:
        try:
            full_path = os.path.join(folder_path, filename)
            doc = docx.Document(full_path)
            # 提取所有段落文字
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
        # 依段落切分，過濾空行
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def generate_answer(query, context):
    """調用最新的 Gemini 1.5 Flash 生成答案"""
    prompt = f"""
    你是一位專業的放電機 (EDM) 操作助手。
    請僅根據以下手冊內容回答使用者的問題。如果手冊內容中沒有答案，請說不知道。
    請回答得詳細且專業。
    
    --- 手冊內容 (CONTEXT) ---
    {context}
    ---
    
    使用者的問題：{query}

    回答（繁體中文）：
    """
    try:
        # 使用 models/ 前綴確保路徑正確，並改用 1.5-flash
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 產生答案時發生錯誤：{e}"

# --- 初始化流程 ---

# 設定頁面資訊
st.set_page_config(page_title="放電機 AI 助理", page_icon="⚡")
st.title("⚡ 放電機操作 AI 小幫手")
st.caption("本系統已內建 AI 授權，並自動讀取目錄下的 Word 操作手冊。")

if 'initialized' not in st.session_state:
    with st.spinner("系統正在初始化文件與向量資料庫，請稍候..."):
        # 1. 載入模型
        st.session_state.model = load_embedding_model()
        
        # 2. 讀取文件 (app.py 所在目錄)
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("錯誤：找不到任何可讀取的 .docx 檔案！請確認手冊已上傳。")
            st.stop()
            
        # 3. 建立向量資料庫 (RAG 核心)
        chunks, _ = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks, show_progress_bar=False)
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        
        st.session_state.initialized = True
        st.success(f"✅ 成功初始化！已載入 {len(doc_names)} 份手冊。")

# --- 使用者對話介面 ---

st.divider()
query = st.text_input("請輸入您的操作問題：", placeholder="例如：如何設定極間電壓？")

if st.button("詢問 AI 小幫手"):
    if query:
        with st.spinner("正在搜尋手冊並分析答案..."):
            # A. 向量檢索 (找出最相關的 5 個片段)
            query_embedding = st.session_state.model.encode([query])
            distances, indices = st.session_state.faiss_index.search(query_embedding, 5)
            
            # 組合上下文
            retrieved_chunks = [st.session_state.chunks[i] for i in indices[0] if i != -1]
            context = "\n\n".join(retrieved_chunks)
            
            # B. 生成答案
            answer = generate_answer(query, context)
            
            # C. 顯示結果
            st.markdown("### 🤖 AI 的回答：")
            st.success(answer)
            
            # D. 提供來源查閱 (增加透明度)
            with st.expander("🔍 查看參考的手冊來源段落"):
                for idx, chunk in enumerate(retrieved_chunks):
                    st.info(f"來源片段 {idx+1}:\n{chunk}")
    else:
        st.warning("請先輸入您的問題。")