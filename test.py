import argparse
from model import get_llm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Test different LLM models.")
    parser.add_argument(
        "-m", "--model", type=str, required=True,
        help="Specify the model to use: 'openai', 'llama2', or 'breeze'."
    )
    args = parser.parse_args()

    # Get the LLM
    llm_type = args.model
    llm = get_llm(llm_type)

    # Input prompt
    prompt = "Answer only yes or no. Is apple red?"

    # Generate the output
    print("\nGenerating output...\n")
    output = llm(prompt)

    # Display the output
    print("Model Output:")
    print(output)

if __name__ == "__main__":
    main()