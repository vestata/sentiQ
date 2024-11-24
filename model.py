from langchain_huggingface import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai.chat_models import ChatOpenAI
import torch
import os
import config


hf_token = config.HF_TOKEN
# llm_type = "lm_studio"


def get_llm(llm_type):
    if llm_type == "openai":
        print("Using OpenAI  gpt-3.5-turbo")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    elif llm_type == "llama2":
        model_name = "meta-llama/Llama-3.2-1B"
        print(f"loading HuggingFace model:{model_name}...")
    elif llm_type == "breeze":
        model_name = "MediaTek-Research/Breeze-7B-Instruct-v1_0"
        print(f"loaing HuggingFace model:{model_name}...")
    elif llm_type == "lm_studio":
        print("Using LM Studio backend")
        api_base = "http://localhost:1234/v1"
        api_key = "lm-studio"

        return ChatOpenAI(
            temperature=0.8,
            openai_api_base=api_base,
            openai_api_key=api_key,
        )
    else:
        raise ValueError(f"invalid llm_type:{llm_type}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map={"": "cuda"},
        token=hf_token,
    )

    hf_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        truncation=True,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

    return HuggingFacePipeline(pipeline=hf_pipeline)
