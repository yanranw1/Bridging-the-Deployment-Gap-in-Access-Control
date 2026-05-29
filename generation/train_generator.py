import os


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from datasets import load_dataset
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
    TrainingArguments,
    set_seed
)
from peft import LoraConfig
import torch
from trl import SFTTrainer
import click
from pathlib import Path

print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    
set_seed(1)


def split_train_test(dataset, test_size=0.2):
    splits = dataset.train_test_split(test_size=test_size)
    return splits['train'], splits['test']


def preprocess(dataset, eos):
    combined = []
    for ex in dataset:
        combined.append(f"Policy: {ex['input']}\nEntities: {ex['output']}{eos}")

    dataset = dataset.add_column('text', combined)

    return dataset


def train_generator(dataset_path, num_steps=500, learning_rate=2e-4, batch_size=8, lora_alpha=32, lora_dropout=0.05,
                    lora_r=16, out_dir="checkpoints/acp_phi_2_t2p", val=True):
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "meta-llama/Meta-Llama-3-8B"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        offload_folder="offload",
        low_cpu_mem_usage=True
    )

    model.config.pretraining_tp = 1
    model.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        # target_modules='all-linear' #change
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    )


    acp_dataset = load_dataset('csv', data_files=dataset_path, split='train')
    acp_dataset = preprocess(acp_dataset, tokenizer.eos_token)

    if val:
        acp_train_dataset, acp_val_dataset = split_train_test(acp_dataset, 0.2)
    else:
        acp_train_dataset, acp_val_dataset = acp_dataset, None

    if acp_val_dataset != None:

        training_arguments = TrainingArguments(
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant':True},
            output_dir=out_dir,
            per_device_train_batch_size=1,#changed from batch_size to 1
            gradient_accumulation_steps=4, #changed from batch_size to 4
            evaluation_strategy='steps',
            optim="paged_adamw_32bit",
            save_steps=10,
            logging_steps=10,
            eval_steps=10,
            do_eval=True,  # Change
            learning_rate=learning_rate,
            max_grad_norm=0.3,
            max_steps=num_steps,
            warmup_ratio=0.03,
            group_by_length=False,
            lr_scheduler_type='constant',
            report_to='none'
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=acp_train_dataset,
            eval_dataset=acp_val_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )

    else:

        training_arguments = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=batch_size,
            optim="paged_adamw_32bit",
            save_steps=10,
	        gradient_checkpointing = True,
	        gradient_checkpointing_kwargs={'use_reentrant':True},
            logging_steps=10,
            do_eval=False,
            learning_rate=learning_rate,
            bf16=True,
            max_grad_norm=0.3,
            max_steps=num_steps,
            warmup_ratio=0.03,
            group_by_length=True,
            lr_scheduler_type='constant',
            report_to='none'
        )
        trainer = SFTTrainer(
            model=model,
            train_dataset=acp_train_dataset,
            peft_config=peft_config,
            dataset_text_field="text",
            max_seq_length=512,
            tokenizer=tokenizer,
            args=training_arguments,
            packing=False
        )

    trainer.train()


@click.command()
@click.option('--dataset_path',
              help='Location of the dataset',
              show_default=True,
              default="combined_verification_acps_missing_rules_phi_errorids_train_250.csv",
              required=True)
@click.option('--num_steps', default=500, help='Number of steps to train', show_default=True)
@click.option('--learning_rate', default=2e-4, help='Learning rate', show_default=True)
@click.option('--batch_size', default=1, help='Batch size', show_default=True) #changed from 4 to 1
@click.option('--lora_alpha', default=32, help='LoRA alpha', show_default=True)
@click.option('--lora_dropout', default=0.05, help='LoRA Dropout rate', show_default=True)
@click.option('--lora_r', default=16, help='LoRA rank. For best practices this should be 0.5*lora_alpha',
              show_default=True)
@click.option('--val', default=True, help='Validate',
              show_default=True)
@click.option('--out_dir', default='../checkpoints/llama3/ibm', help='Output directory', show_default=True)
def main(dataset_path='../data/overall/train.csv', num_steps=500, learning_rate=2e-4, batch_size=8, lora_alpha=32,
         lora_dropout=0.05, lora_r=16, val=True, out_dir='../checkpoints/generation/train'):
    """Trains the access control policy generation model"""
    print('\n =========================== Trining details =========================== \n')
    print(
        f'Dataset: {dataset_path}\nNum. of steps: {num_steps}\nLearning rate: {learning_rate}\nValidate: {val}\nBatch size: {batch_size}\nCheckpoint dir.: {out_dir}\n')
    print(' ======================================================================= \n')
    train_generator(dataset_path=dataset_path, num_steps=num_steps, learning_rate=learning_rate, batch_size=batch_size, lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout, lora_r=lora_r, out_dir=out_dir, val=val)


if __name__ == '__main__':
    set_seed(1)
    main()
