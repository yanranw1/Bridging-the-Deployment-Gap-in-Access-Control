import logging
from peft import PeftModel
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    BertTokenizerFast,
)
import pandas as pd
from tqdm import tqdm
import click
from utils import verify_policy, generate_policy

logging.basicConfig(
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level="INFO",
)

PATH_CKPT = "demo/"


def load_models_tokenizers(device):
    logging.info("Loading checkpoints ...")
    id_model_name = PATH_CKPT + "checkpoints/checkpoint_identification/"

    id_model = BertForSequenceClassification.from_pretrained(
        id_model_name, num_labels=2
    ).to(device)
    id_tokenizer = BertTokenizerFast.from_pretrained(id_model_name)

    id_model.eval()

    ex_model_name = "meta-llama/Meta-Llama-3-8B"
    peft_model_id = PATH_CKPT + "checkpoints/checkpoint_generation/"

    base_model = AutoModelForCausalLM.from_pretrained(
        ex_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    base_model.config.pretraining_tp = 1
    base_model.config.use_cache = True

    gen_model = PeftModel.from_pretrained(base_model, peft_model_id).to(device)
    gen_tokenizer = AutoTokenizer.from_pretrained(
        ex_model_name, trust_remote_code=True, padding_side="left"
    )
    gen_tokenizer.pad_token = gen_tokenizer.eos_token
    gen_model.eval()

    ver_model_name = "facebook/bart-large"
    verification_ckpt = PATH_CKPT + "checkpoints/checkpoint_verification/"

    ver_tokenizer = AutoTokenizer.from_pretrained(ver_model_name)
    ver_model = AutoModelForSequenceClassification.from_pretrained(
        verification_ckpt
    ).to(device)
    ver_model.eval()

    return id_tokenizer, id_model, gen_tokenizer, gen_model, ver_tokenizer, ver_model


@click.command()
@click.option("--device", default="cuda:0", help="CUDA device", show_default=True)
def main(device):

    logging.info("Loading preprocessed sentences ...")
    with open("sents.json", "r") as f:

        sents = json.load(f)
    id_tokenizer, id_model, gen_tokenizer, gen_model, ver_tokenizer, ver_model = (
        load_models_tokenizers(device)
    )
    inputs, outputs, verifications = [], [], []

    logging.info("Generating and Verifying ...")
    for s in tqdm(sents):
        inputs.append(s)
        output, flag = generate_policy(
            s, id_tokenizer, id_model, gen_tokenizer, gen_model, device
        )
        if flag:
            ver_res, _, _ = verify_policy(s, output, ver_tokenizer, ver_model, device)
            verifications.append(ver_res)
        else:
            verifications.append("None")

        outputs.append(output)

    df = pd.DataFrame(
        {"inputs": inputs, "outputs": outputs, "verifications": verifications}
    )

    logging.info("The results are saved in: results.csv")

    df.to_csv("results.csv")

    print(
        "\n ================================ Completed ================================\n"
    )


if __name__ == "__main__":
    main()
