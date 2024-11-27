from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Route question

# Prompt Template
route_instruction = """
You are a decision-making system responsible for directing user questions to the appropriate tool.
If the question is related to " Suspicious Activity", output 'vectorstore'.
Otherwise, output 'web_search'.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", route_instruction),
        ("human", "{question}"),
    ]
)

# RAG

# Prompt Template
rag_instruction = """
You are an assistant responsible for addressing user questions. Utilize the information extracted from the provided documents to respond to the questions.
If the answer to a question cannot be found within the documents, simply reply that you don't know. Do not fabricate an answer.
Note: Please ensure the accuracy of your answers.
"""

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", rag_instruction),
        ("system", "documents: \n\n {documents}"),
        ("human", "question: {question}"),
    ]
)

# Plain LLM

# Prompt Template
llm_instruction = """
You are an assistant responsible for addressing user questions. Utilize your knowledge to respond to the questions.
When responding to questions, please ensure the accuracy of your answers. Do not fabricate an answer.
"""

llm_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", llm_instruction),
        ("human", "question: {question}"),
    ]
)


# Retrieval Grader

# Prompt Template
relevance_instruction = """
You are an evaluator responsible for assessing the relevance of a document to a user's question.
If the document contains keywords or semantics related to the user's question, rate it as relevant.
Output 'yes' or 'no' to indicate whether the document is relevant to the question.
"""
relevance_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", relevance_instruction),
        ("human", "document: \n\n {document} \n\n question: {question}"),
    ]
)

# =====================================================
# Grader LLM
# =====================================================

# Hallucination Grader

# Prompt Template
hallucination_instruction = """
You are an evaluator responsible for determining if an LLM's response is fabricated.

You will be given a document and the corresponding LLM response. Please output 'yes' or 'no' as your judgment.

'Yes' means the LLM's response is fabricated and not based on the document content. 'No' means the LLM's response is not fabricated and is derived from the document content.
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", hallucination_instruction),
        ("human", "document: \n\n {documents} \n\n LLM response: {generation}"),
    ]
)

# Answer Grader

# Prompt Template
evalution_instruction = """
You are an evaluator responsible for determining if an answer addresses the question.

Output 'yes' or 'no'. 'Yes' means the answer does address the question. 'No' means the answer does not address the question.
"""
# Prompt
evalution_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", evalution_instruction),
        ("human", "User question: \n\n {question} \n\n answer: {generation}"),
    ]
)
