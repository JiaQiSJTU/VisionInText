# encoding = "utf-8"

# Data synthesis:
# step1: Filter samples from the training data that can be correctly recognized by Qwen2.5-VL-7B-Instruct with peft
# step2: Use GPT-5 for data analysis and synthesis via prompting

import os
import json
# from utils.load_dataset import AsciiDataset

from utils.data_utils import encode_image
from qwen_vl_utils import process_vision_info
# import base64
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
)
from utils.prompts import IMAGE_ONLY_PROMPT
from utils.post_processing import postprocess_function_by_txt
from peft import PeftModel
from tqdm import tqdm
import torch
import argparse
from openai import OpenAI


def data_filter(model_dir, peft_model_dir, data_dir, img_dir, output_file):
    '''
    # Filter samples from the training data that can be correctly recognized by the model
    '''

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype="auto", device_map="auto")
    # , min_pixels=min_pixels, max_pixels=max_pixels)
    processor = AutoProcessor.from_pretrained(model_dir)
    if peft_model_dir:
        print("load peft model")
        model = PeftModel.from_pretrained(
            model,
            peft_model_dir,
        )

    data_list = []
    with open(data_dir, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    output_file = open(output_file, "a+", encoding="utf-8")

    for i, example in tqdm(enumerate(data_list)):

        ascii_art = example["ascii_art"]
        choices = "\nA: " + example["choices"][0] \
            + "\nB: " + example["choices"][1] \
            + "\nC: " + example["choices"][2] \
            + "\nD: " + example["choices"][3]
        image_path = example["image_path"]
        labels = ["A", "B", "C", "D"][example["labels"].index(1)]
        ori_choices = str(example["choices"])

        base64_image = encode_image(os.path.join(img_dir, image_path))
        input_txt = IMAGE_ONLY_PROMPT.format(choices=choices)

        # SFT models
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image;base64,{base64_image}"},
                    {"type": "text", "text": input_txt}
                ]
            }
        ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        processed_inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        )
        processed_inputs = processed_inputs.to(model.device)

        with torch.inference_mode(mode=True):
            model_outputs = model.generate(
                **processed_inputs,
                do_sample=True,
                temperature=1.0,
                use_cache=True,
                output_scores=True,
                max_new_tokens=1024,
                return_dict_in_generate=True
            )

            if type(processed_inputs) == dict:
                input_len = processed_inputs["input_ids"].size(1)
            else:
                input_len = processed_inputs.input_ids.size(1)

            sequences = model_outputs.sequences[:, input_len:]
            preds = processor.tokenizer.batch_decode(
                sequences, skip_special_tokens=True)
        acc = postprocess_function_by_txt(preds, [labels], [ori_choices])

        if acc == 1:
            output_file.write(json.dumps(example)+"\n")

    output_file.close()


SYNTHESIZE_PROMPT = """
Please answer the multi-choice question based on the given ASCII art in both image and text formats.

[ASCII ART]
{ascii_art}

[Question]
What is depicted in the above ASCII art?{choices}


Please analyze which parts of the ASCII art text string depict which objects, with reference to the specific content in the text string. Please analyze with the image in mind, but do not mention the image in your analysis. 
Then, answer with the option's letter from the given choices.

Output your answer in the following format:
```json
{{
    "analysis": "your analysis",
    "answer": "your choice"
}}
```
"""


def data_synthesize(model_name, api_key, base_url, data_dir, img_dir, output_file):

    client = OpenAI(
        base_url=base_url,
        api_key=api_key
    )

    data_list = []
    with open(data_dir, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    output_file = open(output_file, "a+", encoding="utf-8")

    for i, example in tqdm(enumerate(data_list)):
        # if i <= 100:
        #     continue

        print(i)
        ascii_art = example["ascii_art"]
        choices = "\nA: " + example["choices"][0] \
            + "\nB: " + example["choices"][1] \
            + "\nC: " + example["choices"][2] \
            + "\nD: " + example["choices"][3]
        image_path = example["image_path"]
        labels = ["A", "B", "C", "D"][example["labels"].index(1)]
        ori_choices = str(example["choices"])

        base64_image = encode_image(os.path.join(img_dir, image_path))
        input_txt = SYNTHESIZE_PROMPT.format(
            ascii_art=ascii_art, choices=choices)

        messages = [{"role": "user", "content": [
            {"type": "text", "text": input_txt},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=1.0,
            response_format={"type": "json_object"}
            # "json_mode"= True
        )
        response = response.choices[0].message.content

        try:
            response = json.loads(response)

            acc = postprocess_function_by_txt(
                response["answer"], [labels], [ori_choices])
            # print(response)
            # exit(0)
            if acc == 1:
                example["analysis"] = response["analysis"]
                example["gpt5-answer"] = response["answer"]
                output_file.write(json.dumps(example)+"\n")
                output_file.flush()
            else:
                print(f"error: {response['answer']}")
        except:
            print(f"error: {response}")
    output_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt-5")
    parser.add_argument("--peft_model_dir", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="./data/train/train_filter.jsonl")
    parser.add_argument("--img_dir", type=str, default="./data/train/img")
    parser.add_argument("--output_file", type=str, default="./data/train/train_synthesize.jsonl")
    parser.add_argument("--do_filter", action="store_true")
    parser.add_argument("--do_synthesize", action="store_true")
    args = parser.parse_args()

    if args.do_filter:
        data_filter(model_dir=args.model_name,  # "./Qwen/Qwen2.5-VL-7B-Instruct",
                    # "./outputs/Qwen2.5-VL-7B-Instruct-image-only-qkv",
                    peft_model_dir=args.peft_model_dir,
                    data_dir=args.data_dir,  # "./data/train/train.jsonl",
                    img_dir=args.img_dir,  # "./data/train/img",
                    output_file=args.output_file  # "./data/train/train_filter.jsonl"
                    )

    if args.do_synthesize:
        data_synthesize(model_name=args.model_name,  # "gpt-5"
                        api_key=args.api_key,
                        base_url=args.base_url,
                        data_dir=args.data_dir,  # "./data/train/train_filter.jsonl"
                        img_dir=args.img_dir,  # "./data/train/img",
                        output_file=args.output_file  # "./data/train/train_synthesize.jsonl"
                        )
