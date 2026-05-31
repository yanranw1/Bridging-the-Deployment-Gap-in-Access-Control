import warnings
warnings.filterwarnings("ignore")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import click
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from model_generation import ModelGeneration
from component_manipulation import ComponentManipulation

import pandas as pd


def generate_dataset(dataset_path, model_dir, num_beams=5, device='cuda:0', save_name='verification_dataset.csv'):

    """Generates the access control policy verification dataset"""

    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    gen_tokenizer = AutoTokenizer.from_pretrained(model_dir)

    gen_model = model.to(device)
    gen_model.eval()

    df = pd.read_csv(dataset_path)

    print('\n========================= Step 1: Model-based generation =========================\n')

    gen_instance = ModelGeneration(df, gen_model, gen_tokenizer, device)
    binary_dataset = gen_instance.get_binary_dataset(num_beams)

    print('\n========================= Step 2: Component Manipulation =========================\n')
    print("$$$$$$$$1")
    manipulation = ComponentManipulation(binary_dataset)
    print("$$$$$$$$2")
    augmented_dataset = manipulation.augment(num_times=num_beams)
    print("$$$$$$$$3")
    augmented_dataset.to_csv(save_name)
    print('\n====================================== Saved ! ===================================\n')


@click.command()
@click.option('--dataset_path', default='/home/ubuntu/agentv-main/email_agent/dataset/combined_test.csv', required=True, show_default=True, help='Location of the train dataset used to train the generator')
@click.option('--model', default='/home/ubuntu/agentv-main/generation/nl_acp_model', required=True, show_default=True, help='Fine-tuned T5 model directory (saved by train_t5.py)')
@click.option('--num_beams', default=5, show_default=True, help='Number of beams')
@click.option('--device', default='cuda:0', show_default=True, help='GPU/CPU')
@click.option('--save_name', default='verification_dataset.csv', show_default=True, help='Name of the final dataset')
def main(dataset_path='../../data/overall/train.csv',
         model='./nl_acp_model',
         num_beams=5, device='cuda:0', save_name='verification_dataset.csv'):

    """Generates the verification dataset"""

    generate_dataset(dataset_path, model, num_beams=num_beams, device=device, save_name=save_name)


if __name__ == '__main__':
    main()