# encoding = "utf-8"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration, 
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
    ChameleonProcessor, ChameleonForConditionalGeneration
)
import json
from tqdm import tqdm
import argparse
from PIL import Image
import os
from peft import PeftModel


from utils.load_dataset import AsciiDataset
from utils.conversations_mm import conv_templates
from utils.post_processing import *

# from yi_llava.conversation import conv_templates as yi_conv_templates
# from yi_llava.mm_utils import (
#     KeywordsStoppingCriteria,
#     expand2square,
#     get_model_name_from_path,
#     load_pretrained_model,
#     tokenizer_image_token,
# )
# from yi_llava.model.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, key_info
from utils.prompts import *


def preprocess_function(examples, processor, model_template, image_dir, mode="text-only",add_special_tokens=True):
    '''
    dict_keys(['url', 'original_category', 'category-1', 'category-2', 
                'category-3', 'reserve', 'style', 'author', 'ascii_art', 
                'original_ascii_art', 'choices', 'labels', 'image_path'])
    '''

    inputs = []
    raw_images = []
    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode=="text-only":
            input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)
            
        elif mode=="image-only":
            input_txt = "<image>\n" + IMAGE_ONLY_PROMPT.format(choices=choices)

        elif mode=="both":
            input_txt = "<image>\n" + TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)
        
        # SFT models
        conv_template = conv_templates[model_template].copy()
        conv_template.append_message(conv_template.roles[0], input_txt)
        conv_template.append_message(conv_template.roles[1], None)
        inputs.append(conv_template.get_prompt())

        if mode!="text-only":
            raw_image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
            raw_images.append(raw_image)
 
    if mode!="text-only":
        processed_inputs = processor(inputs, raw_images, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens) 
    else:
        processed_inputs = processor.tokenizer(inputs, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)

    return processed_inputs, inputs


def cogvlm_preprocess_function(examples, model, tokenizer, image_dir, mode="text-only", model_template="llama-3"):

    '''
    batch_size = 1
    '''

    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode=="text-only":
            input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)
            image = None

            conv_template = conv_templates[model_template].copy()
            conv_template.append_message(conv_template.roles[0], input_txt)
            conv_template.append_message(conv_template.roles[1], None)
            input_txt = conv_template.get_prompt()


        elif mode=="image-only":
            input_txt = IMAGE_ONLY_PROMPT.format(choices=choices)
            image = Image.open(os.path.join(image_dir, image_path)).convert('RGB')

        elif mode=="both":
            input_txt = TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)
            image = Image.open(os.path.join(image_dir, image_path)).convert('RGB')

        else:
            print("Error: mode doesn't exist.")
            exit(0)
                

    if image is None:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=input_txt,
            history=[],
            template_version='base'
        )
    else:
        input_by_model = model.build_conversation_input_ids(
            tokenizer,
            query=input_txt,
            history=[],
            images=[image],
            template_version='vqa'
        )

    inputs = {
        'input_ids': input_by_model['input_ids'].unsqueeze(0).to(model.device),
        'token_type_ids': input_by_model['token_type_ids'].unsqueeze(0).to(model.device),
        'attention_mask': input_by_model['attention_mask'].unsqueeze(0).to(model.device),
        'images': [[input_by_model['images'][0].to(model.device).to(model.dtype)]] if image is not None else None,
        "pad_token_id": 128002,
    }

    return inputs

def qwen_preprocess_function(examples, tokenizer, image_dir, mode="text-only"):
    '''batch_size = 1'''
    
    query = None
    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):
        
        if mode=="text-only":
            prompt = 'Context: Please answer the multi-choice question based on the given ASCII art:\n{ascii_art}\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'

            input_txt = prompt.format(ascii_art=ascii_art, choices=choices)
            image = None

        elif mode=="image-only":
            prompt = '<img>{image}</img>Context: Please answer the multi-choice question based on the given ASCII art image.\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'
            input_txt = prompt.format(image=os.path.join(image_dir, image_path), choices=choices)

        elif mode=="both":
            prompt = '<img>{image}</img>Context: Please answer the multi-choice question based on the given ASCII art in both image and text formats:\n{ascii_art}\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'
            input_txt = prompt.format(image=os.path.join(image_dir, image_path), ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)
        
        query = input_txt

    inputs = tokenizer(query, return_tensors="pt")

    return inputs


def yi_preprocess_function(examples, model, tokenizer, image_processor, image_dir, mode="text-only"):
    '''batch_size = 1'''

    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):
        image_file = os.path.join(image_dir, image_path)

        if mode=="text-only":
            input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)
            
        elif mode=="image-only":
            input_txt = DEFAULT_IMAGE_TOKEN + "\n"+IMAGE_ONLY_PROMPT.format(choices=choices)

        elif mode=="both":
            input_txt = DEFAULT_IMAGE_TOKEN + "\n"  + TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)

    conv = yi_conv_templates["mm_default"].copy()
    conv.append_message(conv.roles[0], input_txt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(model.device)
    )

    image = Image.open(image_file)
    if getattr(model.config, "image_aspect_ratio", None) == "pad":
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean)
        )
    image_tensor = image_processor.preprocess(image, return_tensors="pt")[
        "pixel_values"
    ][0]

    stop_str = conv.sep
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    inputs = {
        'input_ids': input_ids,
        "images" : image_tensor.unsqueeze(0).to(dtype=model.dtype).to(model.device),
        "stopping_criteria": [stopping_criteria]
    }
    
    return inputs

def chameleon_preprocess_function(examples, processor, image_dir, mode="text-only"):
    
    inputs = []
    raw_images = []
    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode=="text-only":
            input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)
            
        elif mode=="image-only":
            input_txt =  """Please answer the multi-choice question based on the given ASCII art image.\n\n[ASCII ART]\n<image>\n\n[Question]\nWhat is depicted in the above ASCII art?{choices}\n\nAnswer with the option's letter from the given choices directly.""".format(choices=choices)

        elif mode=="both":
            input_txt = """Please answer the multi-choice question based on the given ASCII art in both image and text formats.\n\n[ASCII ART Image]\n<image>\n\n[ASCII ART Text]\n{ascii_art}\n\n[Question]\nWhat is depicted in the above ASCII art?{choices}\n\nAnswer with the option's letter from the given choices directly.""".format(ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)
        
        inputs.append(input_txt)

        if mode!="text-only":
            raw_image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
            raw_images.append(raw_image)
    
    if mode!="text-only":
        processed_inputs = processor(inputs, raw_images, return_tensors="pt", padding=True).to(torch.bfloat16)
    else:
        processed_inputs = processor.tokenizer(inputs, return_tensors="pt", padding=True).to(torch.bfloat16)

    return processed_inputs, inputs

def main(args):

    (CurrentProcessor, CurrentModel, model_template) = {
        "llava-1.5-7b-hf": (AutoProcessor, LlavaForConditionalGeneration, "llava_v1"),
        "llava-1.5-13b-hf": (AutoProcessor, LlavaForConditionalGeneration, "llava_v1"),
        "llava-v1.6-34b-hf": (LlavaNextProcessor, LlavaNextForConditionalGeneration, "chatml_direct"),
        "llava-v1.6-mistral-7b-hf": (LlavaNextProcessor, LlavaNextForConditionalGeneration, "mistral_instruct"),
        "llava-v1.6-vicuna-13b-hf": (LlavaNextProcessor, LlavaNextForConditionalGeneration, "llava_v1"),
        "cogvlm2-llama3-chat-19B": (AutoTokenizer, AutoModelForCausalLM, None),
        "cogvlm-chat-hf": (LlamaTokenizer, AutoModelForCausalLM, None),
        "Qwen-VL-Chat": (AutoTokenizer, AutoModelForCausalLM, None),
        "Yi-VL-34B": (None, None, None),
        "Yi-VL-6B": (None, None, None),
        "chameleon-7b": (ChameleonProcessor, ChameleonForConditionalGeneration, None),
        "chameleon-30b": (ChameleonProcessor, ChameleonForConditionalGeneration, None),
    }[args.model_dir.split("/")[-1]]

    if "llava" in args.model_dir.lower():
        processor = CurrentProcessor.from_pretrained(args.model_dir, padding_side="left", use_fast=False) 
        tokenizer = processor.tokenizer
        model = CurrentModel.from_pretrained(args.model_dir, torch_dtype=torch.float16, device_map="auto")

        if args.peft_model_dir:
            print("load peft model")
            model = PeftModel.from_pretrained(
                    model,
                    args.peft_model_dir,
                )
        print("finish model loading")
        
    elif "cogvlm" in args.model_dir.lower():   
        if "cogvlm2" not in args.model_dir.lower():
            tokenizer = CurrentProcessor.from_pretrained(args.model_dir.replace("cogvlm-chat-hf", "vicuna-7b-v1.5"), trust_remote_code=True)
            model = CurrentModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        else:
            tokenizer = CurrentProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
            model = CurrentModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    elif "qwen" in args.model_dir.lower():
        tokenizer = CurrentProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
        model = CurrentModel.from_pretrained(args.model_dir, device_map="sequential", trust_remote_code=True)

    elif "yi" in args.model_dir.lower():
        tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_dir)
        model = model.to(dtype=torch.bfloat16)

    elif "chameleon" in args.model_dir.lower():
        model = ChameleonForConditionalGeneration.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        processor = ChameleonProcessor.from_pretrained(args.model_dir)
        tokenizer = processor.tokenizer

    model.eval()
    output_file = open(args.output_file_path, "a+", encoding="utf-8")

    test_data = AsciiDataset(data_path=args.test_file_path)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

    accuracy = 0
    for batch_data in tqdm(test_dataloader):
        
        if "llava" in args.model_dir.lower():
            add_special_tokens = False
            processed_inputs, non_processed_inputs = preprocess_function(batch_data, processor, model_template, args.image_dir, args.mode, add_special_tokens)
            processed_inputs = processed_inputs.to(model.device)
                
        elif "cogvlm" in args.model_dir.lower():
            if "cogvlm2" in args.model_dir.lower():
                model_template = "llama-3"
            else:
                model_template = "vicuna_v1.1"
            processed_inputs = cogvlm_preprocess_function(batch_data, model, tokenizer, args.image_dir, args.mode, model_template)
        
        elif "qwen" in args.model_dir.lower():
            processed_inputs = qwen_preprocess_function(batch_data, tokenizer, args.image_dir, args.mode)
            processed_inputs = processed_inputs.to(model.device)
        
        elif "yi" in args.model_dir.lower():
            processed_inputs = yi_preprocess_function(batch_data, model, tokenizer, image_processor, args.image_dir, mode=args.mode)
        
        elif "chameleon" in args.model_dir.lower():
            processed_inputs, non_processed_inputs = chameleon_preprocess_function(batch_data, processor, args.image_dir, args.mode)
            processed_inputs = processed_inputs.to(model.device)
        
        with torch.inference_mode(mode=True):
            model_outputs = model.generate(
                                **processed_inputs,
                                do_sample=False,
                                use_cache=True, 
                                output_scores=True,
                                max_new_tokens=32,
                                return_dict_in_generate=True 
                            )
        
        if type(processed_inputs)==dict:
            input_len = processed_inputs["input_ids"].size(1)
        else:
            input_len = processed_inputs.input_ids.size(1)

        sequences = model_outputs.sequences[:, input_len:]
        preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        
        acc = postprocess_function_by_txt(preds, batch_data["labels"], batch_data["ori_choices"])
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
    parser.add_argument("--test_file_path", type=str, default="./data/test/test.jsonl")
    parser.add_argument("--image_dir", type=str, default="./data/test/img")
    parser.add_argument("--model_dir", type=str, default="/Models/llava-v1.5-7b-hf")
    parser.add_argument("--output_file_path", type=str, default="./evaluations/MLLM/llava-v1.5-7b-hf-both.json")
    parser.add_argument("--mode", type=str, default="both", help="text-only, image-only, both")
    parser.add_argument("--peft_model_dir", type=str, default=None)

    args = parser.parse_args()

    torch_device = "cuda" if torch.cuda.is_available else "cpu"
    args.torch_device=torch_device

    main(args)
