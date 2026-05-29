import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import click
from datasets import load_dataset
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoTokenizer,
)
from model_generation import ModelGeneration
from component_manipulation import ComponentManipulation

import os
import pandas as pd


def generate_dataset(dataset_path, peft_model_id, num_beams=5, device='cuda:0', save_name='verification_dataset.csv'):
    
    """Generates the access control policy verification dataset"""


    base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

    base_model.config.pretraining_tp = 1 
    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(base_model, peft_model_id)
    gen_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", trust_remote_code=True, padding_side="left")
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_model = model.to(device)
    gen_model.eval()


    df = pd.read_csv(dataset_path)
    
    print('\n========================= Step 1: Model-based generation =========================\n')
    
    
    gen_instance = ModelGeneration(df, gen_model, gen_tokenizer, device)
    binary_dataset = gen_instance.get_binary_dataset(num_beams)
    
    print('\n========================= Step 2: Component Manipulation =========================\n')
    
    manipulation = ComponentManipulation(binary_dataset)
    augmented_dataset = manipulation.augment(num_times=num_beams)
    augmented_dataset.to_csv(save_name)
    print('\n====================================== Saved ! ===================================\n')


@click.command()
@click.option('--dataset_path', default='../../data/overall/train.csv', required=True, show_default=True, help='Location of the train dataset used to train the generator')
@click.option('--model', default='../../checkpoints/generation/overall/checkpoint', required=True, show_default=True, help='Generation model checkpoint')
@click.option('--num_beams', default=5, show_default=True, help='Number of beams')
@click.option('--device', default='cuda:0', show_default=True, help='GPU/CPU')
@click.option('--save_name', default='verification_dataset.csv', show_default=True, help='Name of the final dataset')
def main(dataset_path = "../../data/overall/train.csv", 
         model = "../../checkpoints/generation/overall/checkpoint", 
         num_beams=5, device='cuda:0', save_name='verification_dataset.csv'):
    
    """Generates the verification dataset"""
    
    generate_dataset(dataset_path, model, num_beams=num_beams, device=device, save_name=save_name)
    
    
if __name__ == '__main__':
    main()
    

    
    
    
    

