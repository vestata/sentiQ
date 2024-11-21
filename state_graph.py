
from langgraph.graph import END, StateGraph
from langchain.schema import Document

from typing_extensions import TypedDict
from typing import List


from prompt import answer_grader, hallucination_grader, llm_chain, question_router
from tools.web_search import web_search_tool
from tools.rag_generate import retrieve, retrieval_grade, rag_generate


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