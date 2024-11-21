from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai.chat_models import ChatOpenAI
import torch

def get_llm(llm_type="openai", model_name="gpt-3.5-turbo"):
    if llm_type == "openai":
        return ChatOpenAI(model=model_name, temperature=0)
    elif llm_type == "huggingface":
        # Initialize a local HuggingFace model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        # Create a pipeline for text generation
        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Return the HuggingFacePipeline LLM
        return HuggingFacePipeline(pipeline=hf_pipeline)
    else:
        raise ValueError(f"Unsupported llm_type: {llm_type}")