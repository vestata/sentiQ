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
from prompt import (
    route_prompt,
    rag_prompt,
    llm_prompt,
    relevance_prompt,
    hallucination_prompt,
    evalution_prompt,
    danger_prompt,
)


# ====================================================
# global parameter (might remove in the future)
# ====================================================
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY

# flag for plain or rag
generate_flag = "plain"
    

def run(question, graph_type, llm_type):
    inputs = {"question": question}

    # Choose LLM
    llm = model.get_llm(llm_type)

    # llm_type = "lm_studio"
    # llm_type = "openai"

    # 建立 retriever
    retriever = rag_dataloader.vectorstore.as_retriever()

    # ====================================================
    # Route LLM without bind_tools
    # ====================================================

    question_router = route_prompt | llm | StrOutputParser()

    # =====================================================
    # LLM & chain
    # =====================================================

    rag_chain = rag_prompt | llm | StrOutputParser()

    """### Plain LLM"""

    # =====================================================
    # LLM & chain
    # =====================================================

    llm_chain = llm_prompt | llm | StrOutputParser()

    # =====================================================
    # Grader LLM
    # =====================================================

    """### Retrieval Grader"""

    retrieval_grader = relevance_prompt | llm | StrOutputParser()

    """### Hallucination Grader"""

    hallucination_grader = hallucination_prompt | llm | StrOutputParser()

    """### Answer Grader"""

    answer_grader = evalution_prompt | llm | StrOutputParser()

    """### Dangerous Judge"""
    danger_judge = danger_prompt | llm | StrOutputParser()

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
        dangerous: str
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

        if "vectorstore" in datasource:
            print("  -ROUTE TO VECTORSTORE-")
            return "retrieve"
        else:
            print("  -ROUTE TO PLAIN LLM-")
            return "plain_generate"

    # May not need this
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
            # if "no" in grade:
            #     print("  -DECISION: GENERATION DOES NOT ADDRESS QUESTION-")
            #     return "not useful"
            # else:
            #     print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            #     return "useful"
            print("  -DECISION: GENERATION ADDRESSES QUESTION-")
            return "useful"

    def danger_judgement_generate(state):
        """
        Determines whether the situation is dangerous.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        print("---DANGEROUS JUDGEMENT---")

        generation = state["generation"]

        response = danger_judge.invoke({"generation": generation})

        result = response.strip().lower()

        # Check the situation is dangerous or not
        judgement = ""
        if "yes" in result:
            print("  -DECISION: THE SITUATION IS DANGEROUS-")
            judgement = "yes"
        else:
            print("  -DECISION: THE SITUATION IS NOT DANGEROUS-")
            judgement = "no"

        return {"dangerous": judgement, "question": question, "generation": generation}

    """## Build Graph"""

    class PlainGraph:
        def __init__(self):
            self.workflow = StateGraph(GraphState)

        def setup_nodes(self):
            # Define the nodes
            self.workflow.add_node("plain_generate", plain_generate)  # llm
            self.workflow.add_node(
                "danger_judgement_generate", danger_judgement_generate
            )
            self.workflow.set_entry_point("plain_generate")

        def setup_graph(self):
            self.workflow.add_edge("plain_generate", "danger_judgement_generate")
            self.workflow.add_edge("danger_judgement_generate", END)

        def compile_workflow(self):
            # Compile the workflow into an app
            return self.workflow.compile()

        def create_app(self):
            # Public method to set up and compile workflow
            self.setup_nodes()
            self.setup_graph()
            return self.compile_workflow()

    class RagGraph:
        def __init__(self):
            self.workflow = StateGraph(GraphState)

        def setup_nodes(self):
            # Define the nodes
            self.workflow.add_node("retrieve", retrieve)  # retrieve
            self.workflow.add_node(
                "retrieval_grade", retrieval_grade
            )  # retrieval grade
            self.workflow.add_node("rag_generate", rag_generate)  # rag
            self.workflow.add_node("plain_generate", plain_generate)  # llm
            self.workflow.add_node(
                "danger_judgement_generate", danger_judgement_generate
            )  # danger judgement

        def setup_graph(self):
            # Build graph and set entry points
            self.workflow.set_conditional_entry_point(
                route_question,
                {
                    "retrieve": "retrieve",
                    "plain_generate": "plain_generate",
                },
            )
            # Set up edges
            self.workflow.add_edge("retrieve", "retrieval_grade")

            # May not need this
            # Conditional edges for retrieval grading
            self.workflow.add_conditional_edges(
                "retrieval_grade",
                route_retrieval,
                {
                    "rag_generate": "rag_generate",
                },
            )

            # Conditional edges for RAG generation grading
            self.workflow.add_conditional_edges(
                "rag_generate",
                grade_rag_generation,
                {
                    "not supported": "rag_generate",  # Hallucinations: re-generate
                    "useful": "danger_judgement_generate",
                },
            )
            self.workflow.add_edge("plain_generate", "danger_judgement_generate")
            self.workflow.add_edge("danger_judgement_generate", END)

        def compile_workflow(self):
            # Compile the workflow into an app
            return self.workflow.compile()

        def create_app(self):
            # Public method to set up and compile workflow
            self.setup_nodes()
            self.setup_graph()
            return self.compile_workflow()

    # Create the plain llm graph
    state_graph_plain = PlainGraph()
    app_plain = state_graph_plain.create_app()

    # Create the RAG graph
    state_graph_rag = RagGraph()
    app_rag = state_graph_rag.create_app()

    """### 實際測試"""
    state_graphs = {
        "rag": state_graph_rag,
        "plain": state_graph_plain,
        "exp": (state_graph_plain, state_graph_rag),
    }

    apps = {"rag": app_rag, "plain": app_plain, "exp": (app_plain, app_rag)}

    # experiment mode (plain and rag)
    if graph_type == "exp":
        exp_modes = ("rag", "plain")
        # record the result output
        output_result = {f"{exp_modes[i]}_generate": [] for i in range(len(exp_modes))}

        for i in range(len(exp_modes)):
            selected_app = apps.get(exp_modes[i])
            for output in selected_app.stream(inputs):
                print("\n")

            output[f"{exp_modes[i]}_generate"] = output["danger_judgement_generate"]
            # append the result in the output list
            output_result[f"{exp_modes[i]}_generate"] = [
                output[f"{exp_modes[i]}_generate"]["dangerous"],
                output[f"{exp_modes[i]}_generate"]["generation"],
            ]
            print(output[f"{exp_modes[i]}_generate"]["generation"])

        # save the outputs to the csv file
        # print("output_result :", output_result)
        save_output_to_csv(question, llm_type, output_result, "exp")

    # single function plain or RAG
    else:
        selected_app = apps.get(graph_type)
        # record the result output
        output_result = {f"{graph_type}_generate": []}
        # output_result = {f"{generate_flag}_generate": []}
        for output in selected_app.stream(inputs):
            print("\n")

        output[f"{graph_type}_generate"] = output["danger_judgement_generate"]
        # output[f"{generate_flag}_generate"] = output["danger_judgement_generate"]
        # Final generation
        output_result[f"{graph_type}_generate"] = [
            output[f"{graph_type}_generate"]["dangerous"],
            output[f"{graph_type}_generate"]["generation"],
        ]
        # output_result[f"{generate_flag}_generate"] = [
        #     output[f"{generate_flag}_generate"]["dangerous"],
        #     output[f"{generate_flag}_generate"]["generation"],
        # ]

        # print(output_result)
        # if "rag_generate" in output.keys():
        #     print(output["rag_generate"]["generation"])
        # elif "plain_generate" in output.keys():
        #     print(output["plain_generate"]["generation"])

        save_output_to_csv(question, llm_type, output_result, graph_type)


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
        "-l",
        "--llm",
        type=str,
        default="openai",
        help="Specify the LLM to use, e.g., 'openai', 'local",
    )
    args = parser.parse_args()

    # Run the main function with the specified graph
    # run("怎麼識別可疑物品?", args.flag)
    # run("怎麼辨識可疑人物?", args.flag)
    # run("太陽是什麼顏色?", args.flag)

    # run("There are serval people holding the guns toward the police.", args.flag, args.llm)
    run("How to identify the suspicious objects?", args.flag, args.llm)
    run("How to identift the suspicious person?", args.flag, args.llm)
    run("What's the color of the sun?", args.flag, args.llm)
