# encoding = "utf-8"
from utils.data_utils import *
from utils.prompts import *
from utils.load_dataset import AsciiDataset
from qwen_vl_utils import process_vision_info
import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from peft import LoraConfig, TaskType
from peft import get_peft_model
from transformers.trainer_pt_utils import LabelSmoother
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import os
import functools
import random
random.seed(0)

from utils.data_utils import encode_image


IGNORE_INDEX = LabelSmoother.ignore_index


# target_modules = ["qkv", "proj"] #"q_proj", "k_proj", "v_proj", "out_proj",
# peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Models/Meta-Llama-3-8B")


@dataclass
class DataArguments:
    conv_template_name: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    data_dir: str = field(default=None, metadata={"help": "directory of the training data."})
    train_filename: str = field(default=None, metadata={"help": "filename of the training data."})

    model_type: str = field(default="LLM", metadata={"help": "LLM or MLLM"})
    mode: str = field(default="random", metadata={"help": "random, text-only, image-only, both"})
    add_analysis: int = field(default=0, metadata={"help": "add_analysis"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu()
                          for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, processor, template_name, model_type, mode="random"):
        self.processor = processor
        self.template_name = template_name
        self.model_type = model_type
        self.mode = mode

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources_len = []
        examples = []

        if self.model_type == "LLM":

            for instance in instances:

                input_txt = TEXT_ONLY_PROMPT.format(
                    ascii_art=instance["ascii_art"], choices=instance["choices"])

                messages = [
                    {"role": "user", "content": input_txt}
                ]
                prompt = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                examples.append(prompt + instance["label"])
                sources_len.append(len(self.processor.tokenize(prompt)))

            processed_inputs = self.processor(
                examples, return_tensors="pt", padding=True, add_special_tokens=False)
            labels = processed_inputs.input_ids.clone()

            labels[labels == self.processor.pad_token_id] = IGNORE_INDEX
            for source_idx, sl in enumerate(sources_len):
                labels[source_idx, :sl] = IGNORE_INDEX

            processed_inputs["labels"] = labels

        elif self.model_type == "MLLM":  # Qwen/Qwen2.5-VL-7B-Instruct

            all_messages = []

            for instance in instances:
                prompt_list = ["image-only", "text-only", "both"]
                if self.mode == "random":
                    prompt_mode = prompt_list[random.randint(0, 2)]
                else:
                    prompt_mode = self.mode

                if prompt_mode == "text-only":
                    input_txt = TEXT_ONLY_PROMPT.format(
                        ascii_art=instance["ascii_art"], choices=instance["choices"])

                elif prompt_mode == "image-only":
                    input_txt = IMAGE_ONLY_PROMPT.format(
                        choices=instance["choices"])

                elif prompt_mode == "both":
                    input_txt = TEXT_IMAGE_PROMPT.format(
                        ascii_art=instance["ascii_art"], choices=instance["choices"])

                else:
                    print("Error: mode doesn't exist.")
                    exit(0)

                if prompt_mode != "text-only":
                    base64_image = encode_image(instance["image_path"])
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image",
                                    "image": f"data:image;base64,{base64_image}"},
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

                source_texts = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
                source_image_inputs, source_video_inputs = process_vision_info(
                    messages)
                source_processed_inputs = self.processor(
                    text=source_texts,
                    images=source_image_inputs,
                    videos=source_video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                sources_len.append(
                    len(source_processed_inputs["input_ids"][0]))

                messages.append(
                    {"role": "assistant", "content": instance["label"]})
                all_messages.append(messages[:])

            whole_texts = [
                self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in all_messages
            ]
            image_inputs, video_inputs = process_vision_info(all_messages)
            processed_inputs = self.processor(
                text=whole_texts,
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )

            labels = processed_inputs.input_ids.clone()

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

            for source_idx, sl in enumerate(sources_len):
                labels[source_idx, :sl] = IGNORE_INDEX

            processed_inputs["labels"] = labels

        return processed_inputs


def preprocess(example, data_dir, add_analysis=False):

    ascii_art = example["ascii_art"]
    choices = example["choices"]
    if add_analysis:
        label = example["labels"] + ": " + eval(example["ori_choices"])[
            ["A", "B", "C", "D"].index(example["labels"])]
        label = example["analysis"] + "\nSo the answer is " + label
    else:
        label = example["labels"] + ": " + eval(example["ori_choices"])[
            ["A", "B", "C", "D"].index(example["labels"])]

    image_path = os.path.join(data_dir, "img", example["image_path"])

    return dict(
        ascii_art=ascii_art,
        choices=choices,
        label=label,
        image_path=image_path
    )


def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # print(data_args)
    train_dataset = AsciiDataset(os.path.join(
        data_args.data_dir, data_args.train_filename))
    print(data_args.add_analysis == 1)
    partial_function = functools.partial(
        preprocess, data_dir=data_args.data_dir, add_analysis=(data_args.add_analysis == 1))
    train_dataset.map(partial_function)

    data_collator = DataCollatorForSupervisedDataset(
        processor=processor, template_name=data_args.conv_template_name, model_type=data_args.model_type, mode=data_args.mode)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


def train():

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns = False

    if data_args.model_type == "LLM":
        CurrentModel = AutoModelForCausalLM
        CurrentProcessor = AutoTokenizer

    elif data_args.model_type == "MLLM":
        CurrentModel = Qwen2_5_VLForConditionalGeneration
        CurrentProcessor = AutoProcessor

    model = CurrentModel.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # max_memory = {0:"70GiB", 1: "30GiB"},
        # device_map="sequential"
    )

    # Print the model structure to check the specific module names before applying
    # def print_model_modules(model):
    #     for name, module in model.named_modules():
    #         if any(key in name for key in ["q_proj", "k_proj", "v_proj", "out_proj", "attn"]):
    #             print(f"Found: {name} - {type(module).__name__}")
    # print_model_modules(model)

    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
    # print(model.print_trainable_parameters())
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"trainable parameters: {name}")
    # exit(0)

    try:
        processor = CurrentProcessor.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=model.config.max_position_embeddings,  # 5120,
            padding_side="right",
            use_fast=False,
        )
        processor.pad_token = processor.eos_token
        tokenizer = processor
    except AttributeError:
        # min_pixels = 8 * 28 * 28
        # max_pixels = 16 * 28 * 28
        processor = CurrentProcessor.from_pretrained(
            model_args.model_name_or_path,
            padding_side="right",
            use_fast=False,
            # min_pixels=min_pixels,
            # max_pixels=max_pixels
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        tokenizer = processor.tokenizer

    data_module = make_supervised_data_module(
        processor=processor, data_args=data_args)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,

    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(
        trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
