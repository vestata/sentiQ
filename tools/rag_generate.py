from rag_dataloader import retriever
from models.grading_models import retrieval_grader

from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

# LLM & chain
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
rag_chain = prompt | llm | StrOutputParser()

# 測試 rag_chain 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# print(generation)

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