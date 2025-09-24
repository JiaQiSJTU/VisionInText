# encoding = "utf-8"
from openai import OpenAI
from time import sleep
import os
import requests

def llm_generator(messages, api_key, model, base_url="https://api.openai.com/v1", max_tokens=5, temperature=0.0):
    
    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
        )
    
    return response.choices[0].message.content

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="API test")
    parser.add_argument("--api_key", type=str, default="", help="API_KEY")
    args = parser.parse_args()
    
    messages = [{'role': 'user', 'content': "Come on! It must be a great day!"}]

    print("test openai")
    results = llm_generator(messages, args.api_key, "gpt-3.5-turbo-16k", max_tokens=5, temperature=1.0)
    print(results)