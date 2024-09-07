import logging
import os
import sys

import json
import math
from dataclasses import dataclass, field
from typing import List, Optional
import argparse
import json
import torch
import transformers
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser,TrainerCallback
from transformers.integrations.deepspeed import (
    is_deepspeed_available,
)
from peft import LoraConfig
from transformers import TrainingArguments
from transformers.trainer_callback import PrinterCallback
from trl import SFTTrainer

from autotrain import logger
from autotrain.trainers.clm import utils
from autotrain.trainers.clm.params import LLMTrainingParams
from optimum.habana import GaudiConfig, GaudiTrainingArguments
from optimum.habana.trl import GaudiSFTTrainer
from optimum.habana.utils import set_seed

@dataclass
class ScriptArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    #output_dir: Optional[str] = field(default="./", metadata={"help": "output directory name"})
    dataset_name: Optional[str] = field(default=None, metadata={"help": "the dataset name"})
    dataset_config: Optional[str] = field(default=None, metadata={"help": "the dataset config name"})
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use peft"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    max_seq_length: Optional[int] = field(default=1024, metadata={"help": "the max sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    use_flash_attention: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use Habana flash attention for fine-tuning."}
    )
    flash_attention_recompute: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable recompute in Habana flash attention for fine-tuning."}
    )
    flash_attention_causal_mask: Optional[bool] = field(
        default=False, metadata={"help": "Whether to enable causal mask in Habana flash attention for fine-tuning."}
    )

    # LoraConfig
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: List[str] = field(
        default_factory=lambda: None,
        metadata={"help": "Target modules for the LoRA method."},
    )

    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )

def parse_args():
    # get training_config.json from the end user
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_config", type=str, required=True)
    return parser.parse_args()
import mlflow
import mlflow.pytorch

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
root_path = os.path.abspath(os.path.join(parent_dir, "..", "..", ".."))


ml_path = os.path.join(root_path,"mlruns")
mlflow.set_tracking_uri(ml_path)
mlflow.set_experiment("clm sft ")

def train(config):
    with mlflow.start_run() as run:
        logger.info("Starting SFT training...")
        parser = HfArgumentParser((ScriptArguments, GaudiTrainingArguments))
        script_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[2]))
        print("script_args", script_args)
        print("training_args", training_args)
        if script_args.use_peft:
            peft_config = LoraConfig(
                r=script_args.lora_r,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                target_modules=script_args.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
        else:
            peft_config = None

        if training_args.group_by_length and script_args.packing:
            raise ValueError("Cannot use both packing and group by length")

        set_seed(training_args.seed)

        def chars_token_ratio(dataset, tokenizer, nb_examples=400):
            """
            Estimate the average number of characters per token in the dataset.
            """
            total_characters, total_tokens = 0, 0
            for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
                text = prepare_sample_text(example)
                total_characters += len(text)
                if tokenizer.is_fast:
                    total_tokens += len(tokenizer(text).tokens())
                else:
                    total_tokens += len(tokenizer.tokenize(text))

            return total_characters / total_tokens

        def prepare_sample_text(example):
            """Prepare the text from a sample of the dataset."""
            text = f"Question: {example['question']}\n\nAnswer: {example['response_j']}"
            return text

        def create_datasets(tokenizer, args, seed=None):

            if args.dataset_name:
                dataset = load_dataset(
                    args.dataset_name,
                    args.dataset_config,
                    split=args.split,
                    token=script_args.token,
                    num_proc=args.num_workers if not args.streaming else None,
                    streaming=args.streaming
                )
            else:
                raise ValueError("No dataset_name")

            if args.streaming:
                logger.info("Loading the dataset in streaming mode")
                valid_data = dataset.take(args.size_valid_set)
                train_data = dataset.skip(args.size_valid_set)
                train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=seed)
            else:
                dataset = dataset.train_test_split(test_size=args.validation_split_percentage * 0.01, seed=seed)
                train_data = dataset["train"]
                valid_data = dataset["test"]

            # Concatenate 'content' from each message into a single string
            train_data = train_data.map(lambda x: {"text": " ".join([msg['content'] for msg in x['messages']])})
            valid_data = valid_data.map(lambda x: {"text": " ".join([msg['content'] for msg in x['messages']])})
            formatting_func=None
            return train_data, valid_data, formatting_func

        low_cpu_mem_usage = True
        if is_deepspeed_available():
            from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

            if is_deepspeed_zero3_enabled():
                low_cpu_mem_usage = False

        base_model = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=low_cpu_mem_usage,
            torch_dtype=torch.bfloat16,
            token=script_args.token,
        )

        base_model.config.use_cache = False
        if not script_args.use_flash_attention and (
            script_args.flash_attention_recompute or script_args.flash_attention_recompute
        ):
            assert "Need to enable use_flash_attention"
        base_model.generation_config.use_flash_attention = script_args.use_flash_attention
        base_model.generation_config.flash_attention_recompute = script_args.flash_attention_recompute
        base_model.generation_config.flash_attention_causal_mask = script_args.flash_attention_causal_mask

        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

        # log_level = training_args.get_process_log_level()
        # transformers.utils.logging.set_verbosity(log_level)
        # transformers.utils.logging.enable_default_handler()
        # transformers.utils.logging.enable_explicit_format()

        train_dataset, eval_dataset, formatting_func = create_datasets(tokenizer, script_args, seed=training_args.seed)

        # Tokenize the concatenated text
        train_dataset = train_dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True, max_length=script_args.max_seq_length), batched=True)
        eval_dataset = eval_dataset.map(lambda x: tokenizer(x['text'], padding=True, truncation=True, max_length=script_args.max_seq_length), batched=True)

        for item in train_dataset.take(1): 
            print(type(item))  
            print(item)
            
        gaudi_config = GaudiConfig()
        gaudi_config.use_fused_adam = True
        gaudi_config.use_fused_clip_norm = True
        if training_args.do_train:
            trainer = GaudiSFTTrainer(
                model=base_model,
                gaudi_config=gaudi_config,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                dataset_text_field="messages",
                peft_config=peft_config,
                packing=False,
                max_seq_length=script_args.max_seq_length,
                tokenizer=tokenizer,
                args=training_args,
                formatting_func=formatting_func
            )
            train_result = trainer.train()
            trainer.save_model(training_args.output_dir)
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate()
            if isinstance(eval_dataset, torch.utils.data.IterableDataset):
                eval_dataset = list(eval_dataset)

            metrics["eval_samples"] = len(eval_dataset)

            try:
                perplexity = math.exp(metrics["eval_loss"])
            except OverflowError:
                perplexity = float("inf")
            metrics["perplexity"] = perplexity

            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)
from copy import deepcopy
#unified call back
class UnifiedLoggingCallback(TrainerCallback):
    
    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer
        # script_dir = os.path.dirname(os.path.abspath(__file__))
        # parent_dir = os.path.dirname(script_dir)
        # root_path = os.path.abspath(os.path.join(parent_dir, "..", "..", ".."))
        path = os.path.join(root_path,'model_metrics.log')
        logger.add(path, format="{time} | {level} | {message}", level="INFO")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs and 'grad_norm' in logs and 'learning_rate' in logs:
            # Log to Loguru
                loguru_metrics = {
                    "epoch":logs.get("epoch","N/A"),
                    "loss": logs.get("loss", "N/A"),
                    "grad_norm": logs.get("grad_norm", "N/A"),
                    "learning_rate": logs.get("learning_rate", "N/A"),
                }
                logger.info(f"{loguru_metrics}")
            
            # Log to MLflow
                for key, value in loguru_metrics.items():
                    
                    mlflow.log_metric(f"{key}", value, step=state.global_step)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_copy = deepcopy(control)
            
            # Evaluate on the training dataset
            train_metrics = self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
            # Evaluate on the evaluation dataset
            eval_metrics = self._trainer.evaluate(eval_dataset=self._trainer.eval_dataset, metric_key_prefix="eval")
            
            # Combine and log metrics to Loguru
            combined_metrics = {
                "epoch": state.epoch,
                "train_loss": train_metrics.get("train_loss", "N/A"),
                "train_runtime": train_metrics.get("train_runtime", "N/A"),
                "train_samples_per_second": train_metrics.get("train_samples_per_second", "N/A"),
                "train_steps_per_second": train_metrics.get("train_steps_per_second", "N/A"),
                "eval_loss": eval_metrics.get("eval_loss", "N/A"),
                "eval_runtime": eval_metrics.get("eval_runtime", "N/A"),
                "eval_samples_per_second": eval_metrics.get("eval_samples_per_second", "N/A"),
                "eval_steps_per_second": eval_metrics.get("eval_steps_per_second", "N/A"),
                "batch_size": args.per_device_train_batch_size,
                "max_memory_allocated_GB": train_metrics.get("max_memory_allocated (GB)", "N/A"),
            }
            logger.info(f"{combined_metrics}")
            
            # Log metrics to MLflow
            for key, value in train_metrics.items():
                if key == 'memory_allocated (GB)':
                    continue
                elif key == 'total_memory_available (GB)':
                    continue
                elif key == 'max_memory_allocated (GB)': 
                    key = 'max_memory_allocated_GB'
                mlflow.log_metric(f"{key}", value, step=state.global_step)
                
            for key, value in eval_metrics.items():
                if key == 'memory_allocated (GB)':
                    continue
                elif key == 'total_memory_available (GB)':
                    continue
                elif key == 'max_memory_allocated (GB)': 
                    continue
                mlflow.log_metric(f"{key}", value, step=state.global_step)
                
            return control_copy
        else:
            logger.info(f"Epoch {state.epoch} ended without evaluation.")



args = parse_args()
training_config = json.load(open(args.training_config))
train(training_config)