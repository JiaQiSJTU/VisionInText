# encoding = "utf-8"
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import argparse

from peft import PeftModel

from utils.load_dataset import AsciiDataset
from utils.conversations import get_conv_template
from utils.post_processing import *
from utils.prompts import TEXT_ONLY_PROMPT

def preprocess_function(examples, tokenizer, model_template):
    '''
    dict_keys(['url', 'original_category', 'category-1', 'category-2', 
                'category-3', 'reserve', 'style', 'author', 'ascii_art', 
                'original_ascii_art', 'choices', 'labels', 'image_path'])
    '''

    inputs = []
    for ascii_art, choices in zip(examples["ascii_art"], examples["choices"]):
        # print(choices)
        input_txt = TEXT_ONLY_PROMPT
        input_txt = input_txt.format(ascii_art=ascii_art, choices=choices)
        
        if model_template!=None: # SFT models
            conv_template = get_conv_template(model_template)
            conv_template.append_message(conv_template.roles[0], input_txt)
            conv_template.append_message(conv_template.roles[1], None)
            inputs.append(conv_template.get_prompt())

        else: # base models
            inputs.append(input_txt)

    tokenized_inputs = tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=(model_template==None))
    return tokenized_inputs, inputs


def main(args):

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, padding_side="left", use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    if "gemma" in args.model_dir.lower():
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation='eager', device_map="auto")
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, use_safetensors=True, torch_dtype=torch.bfloat16, device_map="auto")
        except:
            model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")

    if args.peft_model_dir:
        print("load peft model")
        model = PeftModel.from_pretrained(
                model,
                args.peft_model_dir,
            )
    print("finish model loading")
    model.eval()

    output_file = open(args.output_file_path, "a+", encoding="utf-8")

    test_data = AsciiDataset(data_path=args.test_file_path)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

    accuracy = 0
    for batch_data in tqdm(test_dataloader):
        tokenized_inputs, non_tokenized_inputs = preprocess_function(batch_data, tokenizer, args.model_template)
        tokenized_inputs = tokenized_inputs.to(model.device)

        with torch.inference_mode(mode=True):
            model_outputs = model.generate(**tokenized_inputs,
                                    do_sample=False,
                                    use_cache=True, 
                                    output_scores=True,
                                    max_new_tokens=32,
                                    return_dict_in_generate=True)
        
        sequences = model_outputs.sequences[:, tokenized_inputs.input_ids.size(1):]
        preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        acc = postprocess_function_by_txt(preds, batch_data["labels"], batch_data["ori_choices"])
        # acc, preds = postprocess_function_by_prob(model_outputs, tokenizer, batch_data["labels"])
        accuracy += acc
        
        assert len(preds) == len(batch_data["labels"])
        for idx in range(len(preds)):
            output_file.write(json.dumps({"pred":preds[idx], "label":batch_data["labels"][idx]})+"\n")

    print(accuracy/len(test_data))
    # assert len(targets)==len(predictions)
    output_file.close()
    return

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="simulation conversation collection")
    parser.add_argument("--test_file_path", type=str, default="./test_set_all/test.jsonl")
    parser.add_argument("--model_dir", type=str, default="/Models/vicuna-7b-v1.5")
    parser.add_argument("--model_template", type=str, default="alpaca") # Yi-34b-chat
    parser.add_argument("--output_file_path", type=str, default="./evaluations/text/tmp.json")
    parser.add_argument("--peft_model_dir", type=str, default=None)
    args = parser.parse_args()

    torch_device = "cuda" if torch.cuda.is_available else "cpu"
    args.torch_device=torch_device

    main(args)