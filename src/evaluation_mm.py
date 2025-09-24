# encoding = "utf-8"
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoProcessor, LlavaForConditionalGeneration,
    LlavaNextProcessor, LlavaNextForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,
    ChameleonProcessor, ChameleonForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration, AutoModel, AutoTokenizer, AutoConfig
)
import json
from tqdm import tqdm
import argparse
from PIL import Image
import os
from peft import PeftModel
from qwen_vl_utils import process_vision_info
import base64

# internVL-3
import math
import numpy as np
import torch
import torchvision.transforms as T
# from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode


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
from utils.data_utils import encode_image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size),
                 interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(
        image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def preprocess_function(examples, processor, model_template, image_dir, mode="text-only", add_special_tokens=True):
    '''
    dict_keys(['url', 'original_category', 'category-1', 'category-2', 
                'category-3', 'reserve', 'style', 'author', 'ascii_art', 
                'original_ascii_art', 'choices', 'labels', 'image_path'])
    '''

    inputs = []
    raw_images = []
    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode == "text-only":
            input_txt = TEXT_ONLY_PROMPT.format(
                ascii_art=ascii_art, choices=choices)

        elif mode == "image-only":
            input_txt = "<image>\n" + IMAGE_ONLY_PROMPT.format(choices=choices)

        elif mode == "both":
            input_txt = "<image>\n" + \
                TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)

        # SFT models
        conv_template = conv_templates[model_template].copy()
        conv_template.append_message(conv_template.roles[0], input_txt)
        conv_template.append_message(conv_template.roles[1], None)
        inputs.append(conv_template.get_prompt())

        if mode != "text-only":
            raw_image = Image.open(os.path.join(
                image_dir, image_path)).convert("RGB")
            raw_images.append(raw_image)

    if mode != "text-only":
        processed_inputs = processor(
            inputs, raw_images, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)
    else:
        processed_inputs = processor.tokenizer(
            inputs, return_tensors="pt", padding=True, add_special_tokens=add_special_tokens)

    return processed_inputs, inputs


def cogvlm_preprocess_function(examples, model, tokenizer, image_dir, mode="text-only", model_template="llama-3"):
    '''
    batch_size = 1
    '''

    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode == "text-only":
            input_txt = TEXT_ONLY_PROMPT.format(
                ascii_art=ascii_art, choices=choices)
            image = None

            conv_template = conv_templates[model_template].copy()
            conv_template.append_message(conv_template.roles[0], input_txt)
            conv_template.append_message(conv_template.roles[1], None)
            input_txt = conv_template.get_prompt()

        elif mode == "image-only":
            input_txt = IMAGE_ONLY_PROMPT.format(choices=choices)
            image = Image.open(os.path.join(
                image_dir, image_path)).convert('RGB')

        elif mode == "both":
            input_txt = TEXT_IMAGE_PROMPT.format(
                ascii_art=ascii_art, choices=choices)
            image = Image.open(os.path.join(
                image_dir, image_path)).convert('RGB')

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

        if mode == "text-only":
            prompt = 'Context: Please answer the multi-choice question based on the given ASCII art:\n{ascii_art}\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'

            input_txt = prompt.format(ascii_art=ascii_art, choices=choices)
            image = None

        elif mode == "image-only":
            prompt = '<img>{image}</img>Context: Please answer the multi-choice question based on the given ASCII art image.\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'
            input_txt = prompt.format(image=os.path.join(
                image_dir, image_path), choices=choices)

        elif mode == "both":
            prompt = '<img>{image}</img>Context: Please answer the multi-choice question based on the given ASCII art in both image and text formats:\n{ascii_art}\nQuestion: What is depicted in the ASCII art?\nOptions:{choices}\nAnswer with the option\'s letter from the given choices directly.\nAnswer:'
            input_txt = prompt.format(image=os.path.join(
                image_dir, image_path), ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)

        query = input_txt

    inputs = tokenizer(query, return_tensors="pt")

    return inputs


# def yi_preprocess_function(examples, model, tokenizer, image_processor, image_dir, mode="text-only"):
#     '''batch_size = 1'''

#     for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):
#         image_file = os.path.join(image_dir, image_path)

#         if mode=="text-only":
#             input_txt = TEXT_ONLY_PROMPT.format(ascii_art=ascii_art, choices=choices)

#         elif mode=="image-only":
#             input_txt = DEFAULT_IMAGE_TOKEN + "\n"+IMAGE_ONLY_PROMPT.format(choices=choices)

#         elif mode=="both":
#             input_txt = DEFAULT_IMAGE_TOKEN + "\n"  + TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)

#         else:
#             print("Error: mode doesn't exist.")
#             exit(0)

#     conv = yi_conv_templates["mm_default"].copy()
#     conv.append_message(conv.roles[0], input_txt)
#     conv.append_message(conv.roles[1], None)
#     prompt = conv.get_prompt()

#     input_ids = (
#         tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         .unsqueeze(0)
#         .to(model.device)
#     )

#     image = Image.open(image_file)
#     if getattr(model.config, "image_aspect_ratio", None) == "pad":
#         image = expand2square(
#             image, tuple(int(x * 255) for x in image_processor.image_mean)
#         )
#     image_tensor = image_processor.preprocess(image, return_tensors="pt")[
#         "pixel_values"
#     ][0]

#     stop_str = conv.sep
#     keywords = [stop_str]
#     stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

#     inputs = {
#         'input_ids': input_ids,
#         "images" : image_tensor.unsqueeze(0).to(dtype=model.dtype).to(model.device),
#         "stopping_criteria": [stopping_criteria]
#     }

#     return inputs


def qwen2_5_preprocess_function(examples, processor, image_dir, mode="text-only"):

    inputs = []
    image_inputs = None
    video_inputs = None

    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode == "text-only":
            input_txt = TEXT_ONLY_PROMPT.format(
                ascii_art=ascii_art, choices=choices)

        elif mode == "image-only":
            # .replace("What is depicted in the above ASCII art?", "What is depicted in the image?")
            input_txt = IMAGE_ONLY_PROMPT.format(choices=choices)

        elif mode == "both":
            input_txt = TEXT_IMAGE_PROMPT.format(
                ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)

        if mode != "text-only":
            base64_image = encode_image(os.path.join(image_dir, image_path))

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
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": input_txt}
                    ]
                }
            ]

        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs.append(text)
        image_inputs, video_inputs = process_vision_info(messages)

    # print("--------------------------------")
    # print(text)
    # print("--------------------------------")
    # print(image_inputs)
    # print("--------------------------------")
    # exit(0)
    processed_inputs = processor(
        text=inputs,
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )

    return processed_inputs, inputs


def internvl3_preprocess_function(examples, image_dir, mode="text-only"):

    for ascii_art, choices, image_path in zip(examples["ascii_art"], examples["choices"], examples["image_path"]):

        if mode == "text-only":
            input_txt = TEXT_ONLY_PROMPT.format(
                ascii_art=ascii_art, choices=choices)

        elif mode == "image-only":
            input_txt = "<image>\n" + IMAGE_ONLY_PROMPT.format(choices=choices)

        elif mode == "both":
            input_txt = "<image>\n" + \
                TEXT_IMAGE_PROMPT.format(ascii_art=ascii_art, choices=choices)

        else:
            print("Error: mode doesn't exist.")
            exit(0)

        if mode != "text-only":
            pixel_values = load_image(os.path.join(
                image_dir, image_path), max_num=12).to(torch.bfloat16).cuda()
        else:
            pixel_values = None

        question = input_txt

    # print(question)
    # print(pixel_values)
    # exit(0)
    return question, pixel_values


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
        # "Yi-VL-34B": (None, None, None),
        # "Yi-VL-6B": (None, None, None),
        "chameleon-7b": (ChameleonProcessor, ChameleonForConditionalGeneration, None),
        "chameleon-30b": (ChameleonProcessor, ChameleonForConditionalGeneration, None),
        "Qwen2.5-VL-7B-Instruct": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        "Qwen2.5-VL-32B-Instruct": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        "Qwen2.5-VL-72B-Instruct": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        "InternVL3-8B-Instruct": (AutoTokenizer, AutoModel, None),
        "InternVL3-14B-Instruct": (AutoTokenizer, AutoModel, None),
        "InternVL3-38B-Instruct": (AutoTokenizer, AutoModel, None),
        "InternVL3-78B-Instruct": (AutoTokenizer, AutoModel, None),
        "Qwen2.5-VL-7B-Instruct-image-only-full": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        "Qwen2.5-VL-7B-Instruct-both-full": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        "Qwen2.5-VL-7B-Instruct-random-full": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        # "Qwen2.5-VL-7B-Instruct-text-only": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        # "Qwen2.5-VL-7B-Instruct-both": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
        # "Qwen2.5-VL-7B-Instruct-random": (AutoProcessor, Qwen2_5_VLForConditionalGeneration, None),
    }[args.model_dir.split("/")[-1]]

    if "llava" in args.model_dir.lower():
        processor = CurrentProcessor.from_pretrained(
            args.model_dir, padding_side="left", use_fast=False)
        tokenizer = processor.tokenizer
        model = CurrentModel.from_pretrained(
            args.model_dir, torch_dtype=torch.float16, device_map="auto")

        if args.peft_model_dir:
            print("load peft model")
            model = PeftModel.from_pretrained(
                model,
                args.peft_model_dir,
            )
        print("finish model loading")

    elif "cogvlm" in args.model_dir.lower():
        if "cogvlm2" not in args.model_dir.lower():
            tokenizer = CurrentProcessor.from_pretrained(args.model_dir.replace(
                "cogvlm-chat-hf", "vicuna-7b-v1.5"), trust_remote_code=True)
            model = CurrentModel.from_pretrained(
                args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
        else:
            tokenizer = CurrentProcessor.from_pretrained(
                args.model_dir, trust_remote_code=True)
            model = CurrentModel.from_pretrained(
                args.model_dir, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

    elif "qwen-" in args.model_dir.lower():
        tokenizer = CurrentProcessor.from_pretrained(
            args.model_dir, trust_remote_code=True)
        model = CurrentModel.from_pretrained(
            args.model_dir, device_map="sequential", trust_remote_code=True)

    # elif "yi" in args.model_dir.lower():
    #     tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_dir)
    #     model = model.to(dtype=torch.bfloat16)

    elif "chameleon" in args.model_dir.lower():
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model_dir, torch_dtype=torch.bfloat16, device_map="auto")
        processor = ChameleonProcessor.from_pretrained(args.model_dir)
        tokenizer = processor.tokenizer

    elif "qwen2.5" in args.model_dir.lower():
        # min_pixels = 256 * 28 * 28
        # max_pixels = 1280 * 28 * 28
        model = CurrentModel.from_pretrained(
            args.model_dir, torch_dtype="auto", device_map="auto")
        # , min_pixels=min_pixels, max_pixels=max_pixels)
        processor = AutoProcessor.from_pretrained(args.model_dir)
        tokenizer = processor
        if args.peft_model_dir:
            print("load peft model")
            model = PeftModel.from_pretrained(
                model,
                args.peft_model_dir,
            )
    elif "internvl3" in args.model_dir.lower():
        device_map = split_model(args.model_dir)
        model = AutoModel.from_pretrained(args.model_dir, torch_dtype=torch.bfloat16, load_in_8bit=False,
                                          low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True, device_map=device_map)
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_dir, trust_remote_code=True, use_fast=False)

    else:
        print("Unsupported model")
        exit(0)

    model.eval()
    output_file = open(args.output_file_path, "a+", encoding="utf-8")

    test_data = AsciiDataset(data_path=args.test_file_path)
    test_dataloader = DataLoader(test_data, shuffle=False, batch_size=1)

    accuracy = 0
    for batch_data in tqdm(test_dataloader):

        if "llava" in args.model_dir.lower():
            add_special_tokens = False
            processed_inputs, non_processed_inputs = preprocess_function(
                batch_data, processor, model_template, args.image_dir, args.mode, add_special_tokens)
            processed_inputs = processed_inputs.to(model.device)

        elif "cogvlm" in args.model_dir.lower():
            if "cogvlm2" in args.model_dir.lower():
                model_template = "llama-3"
            else:
                model_template = "vicuna_v1.1"
            processed_inputs = cogvlm_preprocess_function(
                batch_data, model, tokenizer, args.image_dir, args.mode, model_template)

        elif "qwen-" in args.model_dir.lower():
            processed_inputs = qwen_preprocess_function(
                batch_data, tokenizer, args.image_dir, args.mode)
            processed_inputs = processed_inputs.to(model.device)

        # elif "yi" in args.model_dir.lower():
        #     processed_inputs = yi_preprocess_function(batch_data, model, tokenizer, image_processor, args.image_dir, mode=args.mode)

        elif "qwen2.5" in args.model_dir.lower():
            processed_inputs, non_processed_inputs = qwen2_5_preprocess_function(
                batch_data, processor, args.image_dir, args.mode)
            processed_inputs = processed_inputs.to(model.device)
        elif "internvl3" in args.model_dir.lower():
            question, pixel_values = internvl3_preprocess_function(
                batch_data, args.image_dir, args.mode)

        if "internvl3" not in args.model_dir.lower():

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
            preds = tokenizer.batch_decode(sequences, skip_special_tokens=True)
        else:  # internvl3
            generation_config = dict(
                max_new_tokens=1024, do_sample=True, temperature=1.0)
            response, _ = model.chat(
                tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
            preds = [response]

        acc = postprocess_function_by_txt(
            preds, batch_data["labels"], batch_data["ori_choices"])
        accuracy += acc

        assert len(preds) == len(batch_data["labels"])
        for idx in range(len(preds)):
            output_file.write(json.dumps(
                {"pred": preds[idx], "label": batch_data["labels"][idx]})+"\n")
            output_file.flush()

    print(accuracy/len(test_data))
    # assert len(targets)==len(predictions)

    output_file.close()
    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="simulation conversation collection")
    parser.add_argument("--test_file_path", type=str, default="./data/test/test.jsonl")
    parser.add_argument("--image_dir", type=str, default="./data/test/img")
    parser.add_argument("--model_dir", type=str, default="/Models/llava-v1.5-7b-hf")
    parser.add_argument("--output_file_path", type=str, default="./evaluations/MLLM/llava-v1.5-7b-hf-both.json")
    parser.add_argument("--mode", type=str, default="both",  help="text-only, image-only, both")
    parser.add_argument("--peft_model_dir", type=str, default=None)

    args = parser.parse_args()

    torch_device = "cuda" if torch.cuda.is_available else "cpu"
    args.torch_device = torch_device

    main(args)
