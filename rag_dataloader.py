from pdf2image import convert_from_path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
import pytesseract
import os
import config

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

pdf_path = "data/RAG.pdf"

persist_directory = "vectordb"


if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)


# 檢查 vectordb 目錄是否為空
if (
    os.path.exists(persist_directory)
    and os.path.isdir(persist_directory)
    and os.listdir(persist_directory)
):
    print("--loading from existing vectordb--")
    # 如果 vectordb 有資料，直接從資料庫讀取
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
    )
else:
    print("vectordb is empty, starting PDF OCR and embedding process")

    # 將 PDF 每頁轉為圖片
    loader = PyPDFLoader(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
    doc_split = loader.load_and_split(splitter)

    print("--start embedding--")
    embeddings = OpenAIEmbeddings()

    print("--creating vectorstore and saving to vectordb--")
    # 使用 Chroma 建立 vectorstore，並將其轉為 retriever 型態
    vectorstore = Chroma.from_documents(
        documents=doc_split,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # 在儲存資料後進行保存
    vectorstore.persist()

# 建立 retriever
retriever = vectorstore.as_retriever()
