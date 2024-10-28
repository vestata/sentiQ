from pdf2image import convert_from_path
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import pytesseract
import os


pdf_path = "data/data1.pdf"
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
    images = convert_from_path(pdf_path)

    print("--start OCR--")
    all_text = ""

    # 遍歷每頁圖片並進行 OCR
    for i, image in enumerate(images):
        # 使用 Tesseract OCR 提取文字
        text = pytesseract.image_to_string(
            image, lang="chi_tra"
        )  # 若為繁體中文，可改為 'chi_tra'
        all_text += text + "\n"

    print("--start word split--")
    # 初始化文字分割器
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)

    # 使用文字分割器分割全部文字
    doc_split = splitter.split_text(all_text)

    # 將分割後的文字轉為 Document 物件
    documents = [Document(page_content=chunk) for chunk in doc_split]

    print("--start embedding--")
    # 定義要使用的 Embedding model 將 chunk 內的文字轉為向量
    embeddings = OpenAIEmbeddings()

    print("--creating vectorstore and saving to vectordb--")
    # 使用 Chroma 建立 vectorstore，並將其轉為 retriever 型態
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    # 在儲存資料後進行保存
    vectorstore.persist()

# 建立 retriever
retriever = vectorstore.as_retriever()
