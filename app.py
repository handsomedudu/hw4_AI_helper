import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai
import time

# --- 1. 從 Secrets 安全讀取 API Key ---
if "GOOGLE_API_KEY" in st.secrets:
    api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=api_key)
else:
    st.error("❌ 錯誤：未在 Streamlit Secrets 中找到 GOOGLE_API_KEY。")
    st.stop()

EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

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

def generate_answer(query, context):
    """具備自動切換模型的生成函式"""
    # 嘗試模型優先順序
    model_candidates = ["models/gemini-1.5-flash", "models/gemini-2.0-flash", "models/gemini-pro"]
    
    prompt = f"你是一位專業的放電機助手。請根據以下內容回答問題，若無答案請說不知道：\n\n{context}\n\n問題：{query}\n回答（繁體中文）："

    last_error = ""
    for model_name in model_candidates:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            return response.text, model_name
        except Exception as e:
            last_error = str(e)
            if "429" in last_error:
                # 如果是配額問題，嘗試下一個模型
                continue
            else:
                return f"發生錯誤：{last_error}", model_name

    return f"所有可用模型配額皆已耗盡，請稍後再試。最後錯誤：{last_error}", "None"

# --- 初始化 ---
st.set_page_config(page_title="放電機 AI 助理", page_icon="⚡")
st.title("⚡ 放電機操作 AI 小幫手")

if 'initialized' not in st.session_state:
    with st.spinner("系統分析文件中..."):
        st.session_state.model = load_embedding_model()
        current_folder = os.path.dirname(os.path.abspath(__file__))
        doc_names, doc_texts = load_documents(current_folder)
        
        if not doc_texts:
            st.error("找不到 .docx 檔案！")
            st.stop()
            
        paragraphs = []
        for text in doc_texts:
            paragraphs.extend([p.strip() for p in text.split('\n\n') if p.strip()])
        
        embeddings = st.session_state.model.encode(paragraphs)
        st.session_state.chunks = paragraphs
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        st.session_state.initialized = True

# --- UI ---
query = st.text_input("請輸入操作問題：")
if st.button("詢問 AI"):
    if query:
        with st.spinner("搜尋答案中..."):
            query_embedding = st.session_state.model.encode([query])
            _, indices = st.session_state.faiss_index.search(query_embedding, 5)
            context = "\n\n".join([st.session_state.chunks[i] for i in indices[0]])
            
            answer, used_model = generate_answer(query, context)
            st.markdown(f"### 🤖 AI 的回答 (使用模型: {used_model})")
            st.info(answer)