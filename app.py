import streamlit as st
import os
import docx
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import google.generativeai as genai

# --- Constants ---
# 使用多國語言向量模型，適合處理繁體中文手冊
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'

# --- Functions ---

@st.cache_resource
def load_embedding_model():
    """載入向量模型 (SentenceTransformer)"""
    with st.spinner("正在載入語意分析模型..."):
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
    """讀取資料夾內的 .doc 和 .docx 檔案"""
    doc_texts, doc_names = [], []
    # 過濾暫存檔並讀取 Word 檔案
    files = [f for f in os.listdir(folder_path) if (f.endswith('.docx') or f.endswith('.doc')) and not f.startswith('~$')]
    for filename in files:
        try:
            full_path = os.path.join(folder_path, filename)
            doc = docx.Document(full_path)
            full_text = "\n\n".join([para.text for para in doc.paragraphs if para.text.strip()])
            doc_texts.append(full_text)
            doc_names.append(filename)
        except Exception as e:
            st.error(f"讀取檔案 {filename} 時出錯: {e}")
    return doc_names, doc_texts

def split_text(doc_names, doc_texts):
    """將文件切割成適合 AI 閱讀的段落 (Chunks)"""
    chunks, chunk_sources = [], []
    for i, text in enumerate(doc_texts):
        # 依段落切分，過濾空行
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        chunks.extend(paragraphs)
        chunk_sources.extend([doc_names[i]] * len(paragraphs))
    return chunks, chunk_sources

def search_index(query, model, index, chunks, k=5):
    """在向量索引中搜尋與問題最相關的 5 個段落"""
    query_embedding = model.encode([query])
    distances, indices = index.search(query_embedding, k)
    
    # 取得不重複的索引編號
    unique_indices = list(dict.fromkeys(indices[0]))
    results = [chunks[i] for i in unique_indices if i != -1]
    return results

def generate_answer(query, context):
    """調用 Gemini API 生成回答"""
    prompt = f"""
    你是一位專業的放電機 (EDM) 操作助手。
    請僅根據以下提供的操作手冊內容來回答使用者的問題。
    如果手冊內容中沒有答案，請禮貌地告知你不知道，不要自行編造。
    
    --- 手冊內容 (CONTEXT) ---
    {context}
    --- 結束內容 ---

    使用者的問題：{query}

    請使用繁體中文回答：
    """
    try:
        # 使用最新的 1.5 flash 模型，速度快且支援長文本
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 產生答案時發生錯誤：{e}"

# --- Streamlit 介面佈局 ---

st.set_page_config(page_title="EDM AI Assistant", layout="wide")
st.title("⚡ 放電機操作手冊問答小幫手")
st.caption("基於 Generative AI (RAG) 技術的工業操作輔助系統")

# 側邊欄設定
with st.sidebar:
    st.header("設定 (Settings)")
    # 修正：定義 google_api_key
    google_api_key = st.text_input("輸入 Google API Key", type="password")
    st.markdown("[如何取得 API Key?](https://aistudio.google.com/app/apikey)")
    st.divider()
    st.info("請將 Word 手冊檔案放在與 app.py 相同的目錄下。")

# --- 初始化與資料處理邏輯 ---

def initialize(api_key):
    """啟動時的初始化流程"""
    genai.configure(api_key=api_key)
    
    # 載入模型
    st.session_state.model = load_embedding_model()

    # 讀取當前目錄下的文件
    current_folder = os.path.dirname(os.path.abspath(__file__))
    doc_names, doc_texts = load_documents(current_folder)
    
    if not doc_texts:
        st.error("找不到手冊檔案！請確保 .docx 檔案已上傳。")
        st.stop()
        
    with st.spinner("正在建立語意索引庫..."):
        chunks, chunk_sources = split_text(doc_names, doc_texts)
        embeddings = st.session_state.model.encode(chunks, show_progress_bar=True)
        
        st.session_state.chunks = chunks
        st.session_state.faiss_index = create_faiss_index(np.array(embeddings))
        st.session_state.initialized = True
    
    st.success(f"✅ 成功載入 {len(doc_names)} 份手冊，共 {len(chunks)} 個段落。")

# 檢查 API Key 並執行初始化
if google_api_key:
    if 'initialized' not in st.session_state:
        initialize(google_api_key)
else:
    st.warning("請先在左側輸入 Google API Key 以啟動系統。")
    st.stop()

# --- 主要問答介面 ---

query = st.text_input("請輸入關於放電機操作的問題 (例如：如何設定極間電壓？)", "")

if st.button("開始詢問"):
    if query:
        with st.spinner("搜尋手冊中..."):
            # 1. 檢索相關內容
            retrieved_chunks = search_index(
                query, 
                st.session_state.model, 
                st.session_state.faiss_index, 
                st.session_state.chunks
            )
            context_text = "\n\n".join(retrieved_chunks)

            # 2. 生成回答
            answer = generate_answer(query, context_text)
            
            # 3. 顯示結果
            st.markdown("### 🤖 AI 的回答：")
            st.write(answer)
            
            with st.expander("🔍 查看參考來源"):
                for i, text in enumerate(retrieved_chunks):
                    st.info(f"來源片段 {i+1}:\n{text}")
    else:
        st.warning("請輸入問題內容。")