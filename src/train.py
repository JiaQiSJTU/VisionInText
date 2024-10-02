# encoding = "utf-8"
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
from transformers import AutoModelForCausalLM, AutoTokenizer, LlavaNextProcessor, LlavaNextForConditionalGeneration
from PIL import Image
import os
import functools
import random
random.seed(0)

from utils.load_dataset import AsciiDataset
from utils.conversations import get_conv_template
from utils.conversations_mm import conv_templates
from utils.prompts import *
 
IGNORE_INDEX = LabelSmoother.ignore_index


target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=target_modules)

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

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: str = field(default="adamw_torch")


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
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
        sources = []
        examples = []
        raw_images = []

        if self.model_type == "LLM":

            for instance in instances:
            
                input_txt = TEXT_ONLY_PROMPT.format(ascii_art=instance["ascii_art"], choices=instance["choices"])
                
                conv_template = get_conv_template(self.template_name)
                conv_template.append_message(conv_template.roles[0], input_txt)
                conv_template.append_message(conv_template.roles[1], None)
                prompt = conv_template.get_prompt()
                examples.append(prompt + instance["label"])
                sources.append(prompt)
                
            processed_inputs = self.processor(examples, return_tensors="pt", padding=True, add_special_tokens=False)

            labels = processed_inputs.input_ids.clone()

            labels[labels == self.processor.pad_token_id] = IGNORE_INDEX
            for source_idx, source in enumerate(sources):

                labels[source_idx, :len(self.processor.tokenize(source))] = IGNORE_INDEX
        
        elif self.model_type=="MLLM":
            
            for instance in instances:
                prompt_list = ["image-only", "text-only", "both"]
                if self.mode=="random":
                    prompt_mode = prompt_list[random.randint(0,2)]
                else:
                    prompt_mode = self.mode
                
                if prompt_mode == "text-only":
                    input_txt = TEXT_ONLY_PROMPT.format(ascii_art=instance["ascii_art"], choices=instance["choices"])

                elif prompt_mode == "image-only":
                    input_txt = "<image>\n" + IMAGE_ONLY_PROMPT.format(choices=instance["choices"])

                elif prompt_mode == "both":
                    input_txt = "<image>\n" + TEXT_IMAGE_PROMPT.format(ascii_art=instance["ascii_art"], choices=instance["choices"])

                else:
                    print("Error: mode doesn't exist.")
                    exit(0)

                conv_template = conv_templates[self.template_name].copy()
                conv_template.append_message(conv_template.roles[0], input_txt)
                conv_template.append_message(conv_template.roles[1], None)
                prompt = conv_template.get_prompt()
                examples.append(prompt + instance["label"])
                sources.append(prompt)
                
                if prompt_mode!="text-only":
                    raw_image = Image.open(instance["image_path"]).convert("RGB")
                    raw_images.append(raw_image)

            if prompt_mode!="text-only":
                processed_inputs = self.processor(examples, raw_images, return_tensors="pt", padding=True, add_special_tokens=False) # add "add_special_tokens" in the transformers.models.llava_next
            else:
                processed_inputs = self.processor.tokenizer(examples, return_tensors="pt", padding=True, add_special_tokens=False)

            labels = processed_inputs.input_ids.clone()

            labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX

            for source_idx, source in enumerate(sources):
                labels[source_idx, :len(self.processor.tokenizer.tokenize(source))] = IGNORE_INDEX

        processed_inputs["labels"] = labels

        return processed_inputs

def preprocess(example, data_dir):

    ascii_art = example["ascii_art"]
    choices = example["choices"]
    label = example["labels"] + ": " + eval(example["ori_choices"])[["A", "B", "C", "D"].index(example["labels"])]
    image_path = os.path.join(data_dir, "img", example["image_path"])

    return dict(
        ascii_art = ascii_art,
        choices = choices,
        label = label,
        image_path = image_path
    )

def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    # print(data_args)
    train_dataset = AsciiDataset(os.path.join(data_args.data_dir, data_args.train_filename))
    partial_function = functools.partial(preprocess, data_dir=data_args.data_dir)
    train_dataset.map(partial_function)

    data_collator = DataCollatorForSupervisedDataset(processor = processor, template_name = data_args.conv_template_name, model_type=data_args.model_type, mode=data_args.mode)

    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)

def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.remove_unused_columns=False

    if data_args.model_type == "LLM":
        CurrentModel = AutoModelForCausalLM
        CurrentProcessor = AutoTokenizer

    elif data_args.model_type == "MLLM":
        CurrentModel = LlavaNextForConditionalGeneration
        CurrentProcessor = LlavaNextProcessor

    model = CurrentModel.from_pretrained(
        model_args.model_name_or_path,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        # max_memory = {0:"70GiB", 1: "30GiB"},
        # device_map="sequential"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

        

    try:
        processor = CurrentProcessor.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=model.config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        processor.pad_token = processor.eos_token
        tokenizer = processor
    except AttributeError:
        processor = CurrentProcessor.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=model.config.text_config.max_position_embeddings,
            padding_side="right",
            use_fast=False,
        )
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
        tokenizer = processor.tokenizer

    
    data_module = make_supervised_data_module(processor=processor, data_args=data_args)

    trainer = Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
        
    )
    trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()