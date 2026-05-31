import pandas as pd
from tqdm import tqdm

# All ACP helpers now live in utils.py — import from there so there is a
# single source of truth shared with test_t5.py.
from utils import (
    is_valid_gold,
    is_field_match,
)


class ModelGeneration:
    """
    Wraps the fine-tuned T5 model (loaded externally) and generates
    correct / incorrect ACP beam outputs for each row in a dataset.

    The model and tokenizer are the same objects produced by train_t5.py
    and loaded in verification_dataset.py via AutoModelForSeq2SeqLM /
    AutoTokenizer.  predict() from train_t5.py is NOT called here because
    that helper re-loads the model from disk on every call; instead we
    call model.generate() directly on the already-loaded objects passed
    in at construction time.
    """

    def __init__(self, df: pd.DataFrame, model, tokenizer, device: str):
        self.inputs  = df["input"].to_list()
        self.outputs = df["output"].to_list()

        self.model     = model
        self.tokenizer = tokenizer
        self.device    = device

        # Populated by get_binary_dataset()
        self.inputs_gen:  list = []
        self.outputs_gen: list = []
        self.labels_gen:  list = []

    # ------------------------------------------------------------------
    # Internal generation loop
    # ------------------------------------------------------------------

    def generate(self, num_beams: int = 5):
        """
        For every (input, gold_output) pair run beam search and bucket each
        decoded sequence as correct (field-match) or incorrect.

        Returns
        -------
        ins        : list[str]         – inputs that were not skipped
        corrects   : list[list[str]]   – per-input correct beam outputs
        incorrects : list[list[str]]   – per-input incorrect beam outputs
        """
        ins, corrects, incorrects = [], [], []
        skipped = 0

        for idx_row, (input_str, gold_output) in enumerate(
            tqdm(zip(self.inputs, self.outputs), total=len(self.inputs))
        ):
            # Skip rows with empty or unparseable gold outputs
            if not is_valid_gold(str(gold_output)):
                skipped += 1
                print(f"\n[{idx_row}] SKIP — empty/invalid gold output")
                continue

            # T5 prompt prefix — must match the prefix used in train_t5.py
            input_text = "translate to ACP: " + str(input_str)

            encoded = self.tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            ).to(self.device)

            n_tokens = encoded["input_ids"].shape[-1]
            print(f"\n[{idx_row}] tokens={n_tokens} | {input_text[:100]}...")

            # Skip heavily truncated inputs
            if n_tokens >= 510:
                skipped += 1
                print(f"[{idx_row}] SKIP — input truncated at {n_tokens} tokens")
                continue

            ins.append(input_str)
            c:  list = []   # correct beam outputs
            ic: list = []   # incorrect beam outputs

            try:
                gen_outputs = self.model.generate(
                    **encoded,
                    num_beams=num_beams,
                    num_return_sequences=num_beams,
                    max_new_tokens=256,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            except Exception as e:
                print(f"[{idx_row}] ERROR during generate(): {e} — skipping row")
                ins.pop()
                skipped += 1
                continue

            for beam_idx in range(num_beams):
                try:
                    result = self.tokenizer.decode(
                        gen_outputs[beam_idx], skip_special_tokens=True
                    )
                except Exception as e:
                    print(
                        f"[{idx_row}] ERROR decoding beam {beam_idx}: {e} — skipping beam"
                    )
                    continue

                print(f"  beam {beam_idx}: {result[:80]}")

                try:
                    matched = is_field_match(str(gold_output), result)
                except Exception as e:
                    print(
                        f"[{idx_row}] ERROR in is_field_match beam {beam_idx}: {e}"
                        " — treating as incorrect"
                    )
                    matched = False

                if matched:
                    if result not in c:
                        c.append(result)
                else:
                    if result not in ic:
                        ic.append(result)

            corrects.append(c)
            incorrects.append(ic)

        print(
            f"\nGeneration complete. Skipped {skipped} rows with "
            "empty/invalid gold outputs."
        )
        return ins, corrects, incorrects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_binary_dataset(self, num_beams: int = 5) -> pd.DataFrame:
        """
        Run beam generation and return a DataFrame with columns:
            inputs  – the NL input string
            outputs – a generated ACP string
            labels  – 1 (correct / field-match) or 0 (incorrect)
        """
        ins, corrects, incorrects = self.generate(num_beams)

        for input_str, cl, icl in zip(ins, corrects, incorrects):
            for c in cl:
                self.inputs_gen.append(input_str)
                self.outputs_gen.append(c)
                self.labels_gen.append(1)

            for ic in icl:
                self.inputs_gen.append(input_str)
                self.outputs_gen.append(ic)
                self.labels_gen.append(0)

        return pd.DataFrame(
            {
                "inputs":  self.inputs_gen,
                "outputs": self.outputs_gen,
                "labels":  self.labels_gen,
            }
        )

    def to_csv(self, name: str) -> None:
        """Save the generated dataset to a CSV file."""
        df = pd.DataFrame(
            {
                "inputs":  self.inputs_gen,
                "outputs": self.outputs_gen,  
                "labels":  self.labels_gen,
            }
        )
        df.to_csv(name, index=False)