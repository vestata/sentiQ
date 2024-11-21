from langchain_openai.chat_models import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from models.grading_models import GradeAnswer, GradeDocuments, GradeHallucinations

# Route LLM with tools use
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

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

# structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore, plain_generate])
structured_llm_router = llm.bind_tools(tools=[web_search, vectorstore])

# 使用 LCEL 語法建立 chain
question_router = route_prompt | structured_llm_router

# 測試 Route 功能
# response = question_router.invoke({"question": "東京的經緯度是多少?"})
# print(response.additional_kwargs['tool_calls'])
# response = question_router.invoke({"question": "牙周病與牙齦炎差在哪裡?"})
# print(response.additional_kwargs['tool_calls'])
# response = question_router.invoke({"question": "你好"})
# print('tool_calls' in response.additional_kwargs)

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
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
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

# LLM & chain
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_chain = prompt | llm | StrOutputParser()

# # 測試 llm_chain 功能
# question = "請問為什麼海水是藍的?"
# generation = llm_chain.invoke({"question": question})
# print(generation)


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

# Grader LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_retrieval_grader = llm.with_structured_output(GradeDocuments)

# 使用 LCEL 語法建立 chain
retrieval_grader = grade_prompt | structured_llm_retrieval_grader

# 測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# doc_txt = docs[0].page_content
# response =  retrieval_grader.invoke({"question": question, "document": doc_txt})
# print(response)

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


# Grader LLM
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_hallucination_grader = llm.with_structured_output(GradeHallucinations)

# 使用 LCEL 語法建立 chain
hallucination_grader = hallucination_prompt | structured_llm_hallucination_grader

# 測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# hallucination_grader.invoke({"documents": docs, "generation": generation})

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

# LLM with function call
# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
structured_llm_answer_grader = llm.with_structured_output(GradeAnswer)

# 使用 LCEL 語法建立 chain
answer_grader = answer_prompt | structured_llm_answer_grader

# #測試 grader 功能
# question = "牙周病與牙齦炎差在哪?"
# docs = retriever.invoke(question)
# generation = rag_chain.invoke({"documents": docs, "question": question})
# answer_grader.invoke({"question": question,"generation": generation})