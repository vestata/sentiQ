import argparse
import model

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Select a state graph.")
    parser.add_argument(
        "-f", "--flag", type=str, required=True, 
        help="Specify the graph to use, e.g., 'rag' or 'plain'"
    )
    parser.add_argument(
        "--llm", type=str, default="openai", 
        help="Specify the LLM to use, e.g., 'openai' or 'llama2'"
    )
    args = parser.parse_args()

    llm_type = args.llm
    import main

    # Run the main function with the specified graph
    main.run("怎麼識別可疑物品?", args.flag)
    main.run("怎麼辨識可疑人物?", args.flag)
    main.run("太陽是什麼顏色?", args.flag)