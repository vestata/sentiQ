import os
import config
import argparse
import model
import rag_dataloader

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from langchain.schema import Document
from langgraph.graph import END, StateGraph

from save_to_csv import save_output_to_csv


def save_text_to_file(text, filename):
    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(text)
        print(f"[TEXT SAVED] {filename}")
    except Exception as e:
        print(f"[ERROR] Failed to save text: {e}")


# ====================================================
# global parameter (might remove in the future)
# ====================================================
# from state_graph import WebRagGraph, PlainGraph

os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY

# tmp_type = "lm_studio"
tmp_type = "openai"

# build retriever
retriever = rag_dataloader.vectorstore.as_retriever()

"""## Web Search Tool"""

web_search_tool = TavilySearchResults()


# ====================================================
# Route LLM without bind_tools
# ====================================================

# Prompt Template
instruction = """
You are a decision-making system responsible for directing user questions to the appropriate tool. Your primary goal is to determine whether the input content relates to past incident scenarios and their corresponding solutions stored in a vector database. If the content is describing the scenario, output 'vectorstore'. 
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "{question}"),
    ]
)

llm = model.get_llm(tmp_type)
question_router = route_prompt | llm | StrOutputParser()

# =====================================================
# LLM & chain
# =====================================================
# Prompt Template

instruction = """
You are an assistant responsible for addressing user-described scenarios. Based on the user-described scenario, utilize the corresponding data from the document, including Event Name, Keywords, Incident Description, Response Measures, and News Link information to generate a report.

Make sure to include the original user-described scenario as part of the report for reference at the top of the report.

If the answer to a question cannot be found within the documents, reply with, "I don't know." Do not fabricate any information or answers.

Note: Accuracy is crucial. Always ensure the correctness of your responses based on the provided documents.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("system", "documents: \n\n {documents}"),
        ("human", "question: {question}"),
    ]
)

llm = model.get_llm(tmp_type)
rag_chain = prompt | llm | StrOutputParser()

"""### Plain LLM"""

# Prompt Template
instruction = """
You are an assistant responsible for addressing user-described scenarios. Utilize your knowledge to respond to the questions.

Determine whether the described scenario indicates any potential danger, and if so, generate a detailed report explaining the nature of the danger in this scenario.
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "question: {question}"),
    ]
)

# =====================================================
# LLM & chain
# =====================================================

llm = model.get_llm(tmp_type)
llm_chain = prompt | llm | StrOutputParser()

"""### Retrieval Grader"""

# Prompt Template
instruction = """
You are an evaluator responsible for assessing the relevance of a document to user-described scenarios.

If the document mentions any keywords or semantics related to the user-described scenario, always rate it as relevant.

Output 'yes' to indicate that the document is relevant.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "document: \n\n {document} \n\n question: {question}"),
    ]
)

# =====================================================
# Grader LLM
# =====================================================

llm = model.get_llm(tmp_type)
retrieval_grader = grade_prompt | llm | StrOutputParser()

"""### Hallucination Grader"""

# Prompt Template
instruction = """
You are an evaluator responsible for determining if an LLM's response is fabricated.

You will be given a document and the corresponding LLM response. Please output 'yes' or 'no' as your judgment.

'Yes' means the LLM's response is fabricated and not based on the document content. 'No' means the LLM's response is not fabricated and is derived from the document content.
"""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "document: \n\n {documents} \n\n LLM response: {generation}"),
    ]
)

# =====================================================
# Grader LLM
# =====================================================

llm = model.get_llm(tmp_type)
hallucination_grader = hallucination_prompt | llm | StrOutputParser()

"""### Answer Grader"""

# Prompt Template
instruction = """
You are an evaluator responsible for determining if an answer addresses the question.

Output 'yes' or 'wrong'. 'Yes' means the answer does address the question. 'wrong' means the answer does not address the question.
"""
# Prompt
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", instruction),
        ("human", "User question: \n\n {question} \n\n answer: {generation}"),
    ]
)

# =====================================================
# LLM
# =====================================================

llm = model.get_llm(tmp_type)
answer_grader = answer_prompt | llm | StrOutputParser()

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

    question: str
    generation: str
    documents: List[str]


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

    return {"documents": documents, "question": question}


def web_search(state):
    """
    Web search based on the re-phrased question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with appended web results
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = []

    # Web search
    docs = web_search_tool.invoke({"query": question})
    # web_results = [Document(page_content=d["content"]) for d in docs]
    web_results = [
        Document(page_content=docs[0]["content"]),
        Document(page_content=docs[1]["content"]),
    ]

    documents = documents + web_results

    return {"documents": documents, "question": question}


def retrieval_grade(state):
    """
    Filter retrieved documents based on question.

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
        response = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = response.strip().lower()
        if "yes" in grade:
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
            continue
    return {"documents": filtered_docs, "question": question}


def rag_generate(state):
    """
    Generate answer using vectorstore / web search

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

    generation = llm_chain.invoke({"question": question})
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
    # print("Question Router Output:", source)
    datasource = source.strip().lower()

    if "plain" in datasource:
        print("  -ROUTE TO PLAIN LLM-")
        return "plain_generate"
    else:
        print("  -ROUTE TO VECTORSTORE-")
        return "retrieve"


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
        print(
            "  -DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, ROUTE TO WEB SEARCH-"
        )
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

    response = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = response.strip().lower()

    # Check hallucination
    if "yes" in grade:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY-")
        return "not supported"
    else:
        print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS-")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        response = answer_grader.invoke(
            {"question": question, "generation": generation}
        )
        grade = response.strip().lower()
        # if "wrong" in grade:
        #     print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
        #     return "not useful"
        # else:
        #     print("  -DECISION: GENERATION ADDRESSES QUESTION-")
        return "useful"


"""## Build Graph"""


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
                "retrieve": "retrieve",
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
                "not useful": "web_search",  # Fails to answer question: fallback to web-search
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


class PlainWebRagGraph:
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
                "not useful": "web_search",  # Fails to answer question: fallback to web-search
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

"""test"""
state_graphs = {
    "rag": state_graph_web_rag,
    "plain": state_graph_plain,
    "exp": (state_graph_plain, state_graph_web_rag),
}

apps = {"rag": app_web_rag, "plain": app_plain, "exp": (app_plain, app_web_rag)}


def run(question, graph_flag, time):
    inputs = {"question": question}
    # experiment mode (plain and web+rag)
    if graph_flag == "exp":
        exp_modes = ("rag", "plain")
        # record the result output
        output_result = {f"{exp_modes[i]}_generate": [] for i in range(len(exp_modes))}

        for i in range(len(exp_modes)):
            selected_app = apps.get(exp_modes[i])
            for output in selected_app.stream(inputs):
                print("\n")
            # append the result in the output list
            output_result[f"{exp_modes[i]}_generate"] = output[
                f"{exp_modes[i]}_generate"
            ]["generation"]
            print(output[f"{exp_modes[i]}_generate"]["generation"])

        # save the outputs to the csv file
        # print("output_result :", output_result)
        save_output_to_csv(question, output_result, "exp")

    # single function plain or web+rag
    else:
        selected_app = apps.get(graph_flag)
        # record the result output
        output_result = {f"{graph_flag}_generate": []}
        for output in selected_app.stream(inputs):
            print("\n")

        # Final generation
        output_result[f"{graph_flag}_generate"] = output[f"{graph_flag}_generate"][
            "generation"
        ]
        print(output_result[f"{graph_flag}_generate"])
        # if "rag_generate" in output.keys():
        #     print(output["rag_generate"]["generation"])
        # elif "plain_generate" in output.keys():
        #     print(output["plain_generate"]["generation"])

        save_text_to_file(
            output_result[f"{graph_flag}_generate"], f"history/{time}/report.txt"
        )

        save_output_to_csv(question, output_result, graph_flag)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Select a state graph.")
    parser.add_argument(
        "-f",
        "--flag",
        type=str,
        required=True,
        help="Specify the graph to use, e.g., 'rag', 'plain' or 'exp'",
    )
    parser.add_argument(
        "--llm",
        type=str,
        default="openai",
        help="Specify the LLM to use, e.g., 'openai', 'llama2' or lm_studio",
    )
    args = parser.parse_args()

    llm_type = args.llm

    # Run the main function with the specified graph
    # run("怎麼識別可疑物品?", args.flag)
    # run("怎麼辨識可疑人物?", args.flag)
    # run("太陽是什麼顏色?", args.flag)

    run("attacked classmates with a knife in a classroom", args.flag, "0-0-0")
    # run("How to identift the suspicious person?", args.flag)
    # run("What's the color of the sun?", args.flag)
