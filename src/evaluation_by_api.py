# encoding = "utf-8"
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import base64
import os
import PIL.Image

from utils.load_dataset import AsciiDataset
from utils.conversations import get_conv_template
from utils.backbone_llm import llm_generator
from utils.prompts import TEXT_ONLY_PROMPT, IMAGE_ONLY_PROMPT, TEXT_IMAGE_PROMPT
from utils.post_processing import postprocess_function_by_txt



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def main(args):

    test_data = AsciiDataset(data_path=args.test_file_path)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)
    output_file = open(args.output_file_path, "a+", encoding="utf-8")

    accuracy = 0
    count = -1
    for batch_data in tqdm(test_dataloader):
        count += 1

        ascii_art = batch_data["ascii_art"][0]
        choices = batch_data["choices"][0]
        image_path = os.path.join(args.image_dir, batch_data["image_path"][0])

        if system_prompt and len(system_prompt.strip())!=0:
            messages = [{"role": "system", "content": system_prompt}]
        else:
            messages = []

        if args.mode=="text-only":
            input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)
            messages.append({"role":"user", "content": input_txt})

        elif args.mode=="image-only":
            input_txt = IMAGE_ONLY_PROMPT.format(choices=choices)

            if "gemini" not in args.model_name: # openai
                base64_image = encode_image(image_path)
                messages = [{"role": "user", "content": [
                                {"type": "text", "text": input_txt},
                                {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}]
            else: # gemini
                encoded_image = PIL.Image.open(image_path)
                messages = [{
                    "role":"user", 
                    "content": [encoded_image, input_txt]
                }]

        elif args.mode=="both":
            input_txt = TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)
            
            if "gemini" not in args.model_name: # openai

                base64_image = encode_image(image_path)
                messages = [{"role": "user", "content": [
                                {"type": "text", "text": input_txt},
                                {"type": "image_url","image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                            ]}]
            else: # gemini
                encoded_image = PIL.Image.open(image_path)
                messages = [{
                                "role":"user", 
                                "content": [encoded_image, input_txt]
                            }]
        else:
            print("Mode error")
            exit(0)

        try:
            response = llm_generator(messages = messages[:], api_key=args.api_key,
                                 base_url=args.base_url, temperature=args.temperature,
                                 model=args.model_name, max_tokens=args.max_tokens, mode=args.mode)
        
        except Exception as e:
            print(e)
            response = "None"

        acc = postprocess_function_by_txt([response], batch_data["labels"], batch_data["ori_choices"])
        accuracy += acc

        output_file.write(json.dumps({"pred":response, "label":batch_data["labels"][0]})+"\n")
        # output_file.flush()

    print(accuracy/len(test_data))
    # assert len(targets)==len(predictions)
    output_file.close()
    return

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="evaluation")
    parser.add_argument("--api_key", type=str, default="", help="API_KEY")
    parser.add_argument("--model_name", type=str, default="gemini-1.5-pro-exp-0801", help="testing model") #gpt-4o-2024-05-13
    parser.add_argument("--max_tokens", type=int, default=128, help="The max number of tokens to be generated")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--base_url", type=str, default="https://api.openai.com/v1",) # https://api.together.xyz/v1
    parser.add_argument("--template_name", type=str, default="gemini") # gpt-4o-2024-05-13 #chatgpt
    parser.add_argument("--test_file_path", type=str, default="./data/test/test.jsonl")
    parser.add_argument("--image_dir", type=str, default="./data/test/img")
    parser.add_argument("--output_file_path", type=str, default="gemini-1.5-pro-exp-0801-both.jsonl")
    parser.add_argument("--mode", type=str, default="text-only", help="text-only, image-only, both")
    args = parser.parse_args()

    if args.template_name:
        system_prompt = get_conv_template(args.template_name).system_message # original prompt
    else:
        system_prompt = "You are a helpful assistant."

    args.system_prompt = system_prompt

    main(args)