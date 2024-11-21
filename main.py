import os
import config
import argparse
import model

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_openai.embeddings import OpenAIEmbeddings

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from pdf2image import convert_from_path
import pytesseract

# from state_graph import WebRagGraph, PlainGraph

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ['TAVILY_API_KEY'] = config.TAVILY_API_KEY

"""## Vectorstore"""

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
    embeddings = OpenAIEmbeddings(api_key=config.OPENAI_API_KEY)
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

"""## Web Search Tool"""

web_search_tool = TavilySearchResults()

"""## LLMs

### Question Router
"""

# 定義兩個工具的 DataModel
class web_search(BaseModel):
    """
    網路搜尋工具。若問題與如何識別可疑人物/物品"無關"，請使用web_search工具搜尋解答。
    """
    query: str = Field(description="使用網路搜尋時輸入的問題")

class vectorstore(BaseModel):
    """
    紀錄關於"如何識別可疑人物/物品"的向量資料庫工具。若問題與"如何識別可疑人物/物品"有關，請使用此工具搜尋解答。
    """
    query: str = Field(description="搜尋向量資料庫時輸入的問題")

# class plain_generate(BaseModel):
#     """
#     基本LLM模型輸出，不管任何問題，都會進來給出答案。
#     """
#     query: str = Field(description="使用LLM輸出時輸入的問題")
    

# Prompt Template
instruction = """
你是將使用者問題導向向量資料庫或網路搜尋的專家。
向量資料庫包含有關"如何識別可疑人物/物品"文件。對於這些主題的問題，請使用向量資料庫工具。其他情況則使用網路搜尋工具。"""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "{question}"),
    ]
)

# Route LLM with tools use
llm = model.get_llm()
# structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore, plain_generate])
structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore])


# 使用 LCEL 語法建立 chain
question_router = route_prompt | structured_llm_router

"""### RAG Responder"""

# Prompt Template
instruction = """
你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。
注意：請確保答案的準確性。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("system","文件: \n\n {documents}"),
        ("human","問題: {question}"),
    ]
)

# =====================================================
# LLM & chain
# =====================================================
llm = model.get_llm()
rag_chain = prompt | llm | StrOutputParser()

# 測試 rag_chain 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# print(generation)

"""### Plain LLM"""

# Prompt Teamplate
instruction = """
你是一位負責處理使用者問題的助手，請利用你的知識來回應問題。
回應問題時請確保答案的準確性，勿虛構答案。
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human","問題: {question}"),
    ]
)

# =====================================================
# LLM & chain
# =====================================================

llm = model.get_llm()
llm_chain = prompt | llm | StrOutputParser()

# # 測試 llm_chain 功能
# question = "請問為什麼海水是藍的?"
# generation = llm_chain.invoke({"question": question})
# print(generation)

"""### Retrieval Grader"""

class GradeDocuments(BaseModel):
    """
    確認提取文章與問題是否有關(yes/no)
    """

    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責評估文件與使用者問題的關聯性。
如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
輸出 'yes' or 'no' 代表文件與問題的相關與否。
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}"),
    ]
)

# =====================================================
# Grader LLM
# =====================================================

llm = model.get_llm()
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# 使用 LCEL 語法建立 chain
retrieval_grader = grade_prompt | structured_llm_grader

# 測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# response =  retrieval_grader.invoke({"question": question, "document": doc_txt})
# print(response)

"""### Hallucination Grader"""

class GradeHallucinations(BaseModel):
    """
    確認答案是否為虛構(yes/no)
    """

    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認LLM的回應是否為虛構的。
以下會給你一個文件與相對應的LLM回應，請輸出 'yes' or 'no'做為判斷結果。
'Yes' 代表LLM的回答是虛構的，未基於文件內容 'No' 則代表LLM的回答並未虛構，而是基於文件內容得出。
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}"),
    ]
)


# =====================================================
# Grader LLM
# =====================================================

llm = model.get_llm()
structured_llm_grader = llm.with_structured_output(GradeHallucinations)

# 使用 LCEL 語法建立 chain
hallucination_grader = hallucination_prompt | structured_llm_grader

# 測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# hallucination_grader.invoke({"documents": docs, "generation": generation})

"""### Answer Grader"""

class GradeAnswer(BaseModel):
    """
    確認答案是否可回應問題
    """

    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")

# Prompt Template
instruction = """
你是一個評分的人員，負責確認答案是否回應了問題。
輸出 'yes' or 'no'。 'Yes' 代表答案確實回應了問題， 'No' 則代表答案並未回應問題。
"""
# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}"),
    ]
)

# =====================================================
# LLM with function call
# =====================================================

llm = model.get_llm()
structured_llm_grader = llm.with_structured_output(GradeAnswer)

# 使用 LCEL 語法建立 chain
answer_grader = answer_prompt | structured_llm_grader

# #測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# answer_grader.invoke({"question": question,"generation": generation})

"""# Graph

## Graph state
"""

class GraphState(TypedDict):
    """
    State of graph.

    Attributes:
        question: question
        generation: LLM generation
        documents: list of documents
    """
    question : str
    generation : str
    documents : List[str]

"""## Nodes / Conditional edges"""

def retrieve(state):
    """
    Retrieve documents related to the question.

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)

    return {"documents":documents, "question":question}

def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    # print("This is state: ", state, "\n")
    question = state["question"]
    # documents = state["documents"] if state["documents"] else []
    documents = []

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs]

    documents = documents + web_results

    return {"documents": documents, "question": question}

def retrieval_grade(state):
    """
    filter retrieved documents based on question.

    Args:
        state (dict):  The current state graph

    Returns:
        state (dict): New key added to state, documents, that contains list of related documents.
    """

    # Grade documents
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")

    documents = state["documents"]
    question = state["question"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
            continue
    return {"documents": filtered_docs, "question": question}

def rag_generate(state):
    """
    Generate answer using  vectorstore / web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE IN RAG MODE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"documents": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def plain_generate(state):
    """
    Generate answer using the LLM without vectorstore.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """

    print("---GENERATE PLAIN ANSWER---")
    question = state["question"]
    # print("question: ", question)
    
    generation = llm_chain.invoke({"question": question})
    # print("generation: ", generation)
    return {"question": question, "generation": generation}


### Edges ###
def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    source = question_router.invoke({"question": question})

    # Fallback to plain LLM or raise error if no decision
    if "tool_calls" not in source.additional_kwargs:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
      raise "Router could not decide source"

    # Choose datasource
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == 'web_search':
        print("  -ROUTE TO WEB SEARCH-")
        return "web_search"
    elif datasource == 'vectorstore':
        print("  -ROUTETO VECTORSTORE-")
        return "vectorstore"

def route_retrieval(state):
    """
    Determines whether to generate an answer, or use websearch.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ROUTE RETRIEVAL---")
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        print("  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO WEB SEARCH-")
        return "web_search"
    else:
        # We have relevant documents, so generate answer
        print("  -DECISION: GENERATE WITH RAG LLM-")
        return "rag_generate"

def grade_rag_generation(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    # Check hallucination
    if grade == "no":
        print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS-")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score.binary_score
        if grade == "yes":
            print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            return "useful"
        else:
            print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
            return "not useful"
    else:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-")
        return "not supported"

"""## Build Graph"""

# workflow = StateGraph(GraphState)

# # Define the nodes
# workflow.add_node("web_search", web_search) # web search
# workflow.add_node("retrieve", retrieve) # retrieve
# workflow.add_node("retrieval_grade", retrieval_grade) # retrieval grade
# workflow.add_node("rag_generate", rag_generate) # rag
# workflow.add_node("plain_answer", plain_answer) # llm

# # Build graph
# workflow.set_conditional_entry_point(
#     route_question,
#     {
#         "web_search": "web_search",
#         "vectorstore": "retrieve",
#         "plain_answer": "plain_answer",
#     },
# )
# workflow.add_edge("retrieve", "retrieval_grade")
# workflow.add_edge("web_search", "retrieval_grade")
# workflow.add_conditional_edges(
#     "retrieval_grade",
#     route_retrieval,
#     {
#         "web_search": "web_search",
#         "rag_generate": "rag_generate",
#     },
# )
# workflow.add_conditional_edges(
#     "rag_generate",
#     grade_rag_generation,
#     {
#         "not supported": "rag_generate", # Hallucinations: re-generate
#         "not useful": "web_search", # Fails to answer question: fall-back to web-search
#         "useful": END,
#     },
# )
# workflow.add_edge("plain_answer", END)

# # Compile
# app = workflow.compile()

class PlainGraph:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
    
    def setup_nodes(self):
        # Define the nodes
        self.workflow.add_node("plain_generate", plain_generate)  # llm
        self.workflow.set_entry_point("plain_generate")

    def setup_graph(self):
        self.workflow.add_edge("plain_generate", END)

    def compile_workflow(self):
        # Compile the workflow into an app
        return self.workflow.compile()

    def create_app(self):
        # Public method to set up and compile workflow
        self.setup_nodes()
        self.setup_graph()
        return self.compile_workflow()


class WebRagGraph:
    def __init__(self):
        self.workflow = StateGraph(GraphState)
        
    def setup_nodes(self):
        # Define the nodes
        self.workflow.add_node("web_search", web_search)  # web search
        self.workflow.add_node("retrieve", retrieve)  # retrieve
        self.workflow.add_node("retrieval_grade", retrieval_grade)  # retrieval grade
        self.workflow.add_node("rag_generate", rag_generate)  # rag
        self.workflow.add_node("plain_generate", plain_generate)  # llm

    def setup_graph(self):
        # Build graph and set entry points
        self.workflow.set_conditional_entry_point(
            route_question,
            {
                "web_search": "web_search",
                "vectorstore": "retrieve",
                "plain_generate": "plain_generate",
            },
        )
        # Set up edges
        self.workflow.add_edge("retrieve", "retrieval_grade")
        self.workflow.add_edge("web_search", "retrieval_grade")
        
        # Conditional edges for retrieval grading
        self.workflow.add_conditional_edges(
            "retrieval_grade",
            route_retrieval,
            {
                "web_search": "web_search",
                "rag_generate": "rag_generate",
            },
        )
        
        # Conditional edges for RAG generation grading
        self.workflow.add_conditional_edges(
            "rag_generate",
            grade_rag_generation,
            {
                "not supported": "rag_generate",  # Hallucinations: re-generate
                "not useful": "web_search",       # Fails to answer question: fallback to web-search
                "useful": END,
            },
        )
        self.workflow.add_edge("plain_generate", END)

    def compile_workflow(self):
        # Compile the workflow into an app
        return self.workflow.compile()

    def create_app(self):
        # Public method to set up and compile workflow
        self.setup_nodes()
        self.setup_graph()
        return self.compile_workflow()


# create the web rag graph
state_graph_web_rag = WebRagGraph()
app_web_rag = state_graph_web_rag.create_app()
state_graph_plain = PlainGraph()
app_plain = state_graph_plain.create_app()

"""### 實際測試"""
state_graphs = {
    "rag": state_graph_web_rag,
    "plain": state_graph_plain,
}

apps = {
    "rag": app_web_rag,
    "plain": app_plain,
}

def run(question, graph_flag):
    inputs = {"question": question}
    # select the state graph you want to use
    selected_app = apps.get(graph_flag)
    for output in selected_app.stream(inputs):
        print("\n")

    # Final generation
    if 'rag_generate' in output.keys():
        print(output['rag_generate']['generation'])
    elif 'plain_generate' in output.keys():
        print(output['plain_generate']['generation'])


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Select a state graph.")
    parser.add_argument(
        "-f", "--flag", type=str, required=True, 
        help="Specify the graph to use, e.g., 'rag' or 'plain'"
    )
    parser.add_argument(
        "--llm", type=str, default="openai", 
        help="Specify the LLM to use, e.g., 'openai' or 'llama3'"
    )
    args = parser.parse_args()
    
    # Run the main function with the specified graph
    run("怎麼識別可疑物品?", args.flag)
    run("怎麼辨識可疑人物?", args.flag)
    run("太陽是什麼顏色?", args.flag)