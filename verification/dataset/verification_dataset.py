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
 
import os
import pandas as pd
 
 
def generate_dataset(dataset_path, model_dir, num_beams=5, device='cuda:0', save_name='verification_dataset.csv'):
    
    """Generates the access control policy verification dataset"""
 
 
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    gen_tokenizer = AutoTokenizer.from_pretrained(model_dir)
 
    gen_model = gen_model.to(device)
    gen_model.eval()
 
 
    df = pd.read_csv(dataset_path)
    
    print('\n========================= Step 1: Model-based generation =========================\n')
    
    
    gen_instance = ModelGeneration(df, gen_model, gen_tokenizer, device)
    binary_dataset = gen_instance.get_binary_dataset(num_beams)
    print(binary_dataset)
    
    print('\n========================= Step 2: Component Manipulation =========================\n')
    
    manipulation = ComponentManipulation(binary_dataset)
    augmented_dataset = manipulation.augment(num_times=num_beams)
    augmented_dataset.to_csv(save_name)
    print('\n====================================== Saved ! ===================================\n')

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset_path",
    default="/home/ubuntu/agentv-main/email_agent/dataset/combined_converted.csv",
    required=True,
    show_default=True,
    help="CSV test set used for generation (must have 'input' and 'output' columns)",
)
@click.option(
    "--model",
    default="/home/ubuntu/agentv-main/generation/nl_acp_model-flan-t5",
    required=True,
    show_default=True,
    help="Fine-tuned T5 model directory (saved by train_t5.py)",
)
@click.option("--num_beams", default=5,        show_default=True, help="Number of beams for generation")
@click.option("--device",    default="cuda:0", show_default=True, help="GPU/CPU device string")
@click.option("--save_name", default="verification_dataset.csv", show_default=True, help="Output CSV filename")
def main(dataset_path, model, num_beams, device, save_name):
    """Generate the ACP verification dataset."""
    generate_dataset(
        dataset_path=dataset_path,
        model_dir=model,
        num_beams=num_beams,
        device=device,
        save_name=save_name,
    )


if __name__ == "__main__":
    main()