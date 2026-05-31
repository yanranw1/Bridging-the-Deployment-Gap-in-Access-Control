"""
Verification Dataset Generator
================================
Produces a labelled ACP verification dataset in two steps:

    Step 1 – Model-based generation
        Load the fine-tuned T5 model (trained by train_t5.py) and run beam
        search over the test set.  Each beam output is compared to the gold
        ACP using is_field_match() (from utils.py, the same logic used in
        test_t5.py) and labelled correct (1) or incorrect (0).

    Step 2 – Component manipulation
        Augment the correct predictions with rule-level perturbations
        (swap decision, change subject/action/resource, remove fields, etc.)
        to produce a balanced, diverse verification dataset.

Usage:
    python verification_dataset.py \\
        --dataset_path combined_test.csv \\
        --model        ./nl_acp_model \\
        --num_beams    5 \\
        --device       cuda:0 \\
        --save_name    verification_dataset.csv
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import click
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from model_generation import ModelGeneration
from component_manipulation import ComponentManipulation


def generate_dataset(
    dataset_path: str,
    model_dir: str,
    num_beams: int = 5,
    device: str = "cuda:0",
    save_name: str = "verification_dataset.csv",
) -> None:
    """Generate and save the ACP verification dataset."""

    # ── Load the fine-tuned T5 model (same checkpoint used by train_t5.py) ──
    print(f"Loading model from {model_dir} …")
    model     = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    df = pd.read_csv(dataset_path)

    # ── Step 1: beam-search generation + field-match labelling ──────────────
    print("\n========================= Step 1: Model-based generation =========================\n")
    gen_instance   = ModelGeneration(df, model, tokenizer, device)
    binary_dataset = gen_instance.get_binary_dataset(num_beams)

    # ── Step 2: component-level augmentation ────────────────────────────────
    print("\n========================= Step 2: Component Manipulation =========================\n")
    manipulation      = ComponentManipulation(binary_dataset)
    augmented_dataset = manipulation.augment(num_times=num_beams)

    augmented_dataset.to_csv(save_name, index=False)
    print(f"\n====================================== Saved → {save_name} ===================================\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--dataset_path",
    default="/home/ubuntu/agentv-main/email_agent/dataset/combined_train.csv",
    required=True,
    show_default=True,
    help="CSV test set used for generation (must have 'input' and 'output' columns)",
)
@click.option(
    "--model",
    default="/home/ubuntu/agentv-main/generation/nl_acp_model",
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