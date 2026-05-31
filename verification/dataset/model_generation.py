import pandas as pd
from tqdm import tqdm
import re


# ---------------------------------------------------------------------------
# ACP field-match helpers  (mirrors the logic in the test script)
# ---------------------------------------------------------------------------

def parse_acp(text: str) -> dict:
    """Parse a flat ACP string into a dict of field→value pairs."""
    text = text.strip().lstrip("{").rstrip("}")
    parts = re.split(r';\s*(?=\w+\s*:)', text)
    result = {}
    for part in parts:
        m = re.match(r'\s*(\w+)\s*:\s*(.*)', part.strip(), re.DOTALL)
        if m:
            result[m.group(1).lower().strip()] = m.group(2).strip()
    return result


def extract_resource_ref(resource_val: str):
    """Pull out the id key name and value from a resource field."""
    m = re.search(
        r'(email_id|message_id|msg_id|thread_id|id)\s*=\s*(\S+?)(?:,|;|$)',
        resource_val,
    )
    if m:
        return m.group(1), m.group(2).rstrip(",;")
    return None, None


_ACTION_ALIASES: dict = {
    "reply_email":   {"reply_email",   "send_email"},
    "forward_email": {"forward_email", "send_email"},
}

def _action_normalize(action: str) -> str:
    return re.sub(r"[\s_]", "", action).lower()


def action_match(exp_action: str, pred_action: str) -> bool:
    exp_norm  = _action_normalize(exp_action)
    pred_norm = _action_normalize(pred_action)
    if exp_norm == pred_norm:
        return True
    for canonical, aliases in _ACTION_ALIASES.items():
        if exp_norm == _action_normalize(canonical):
            return pred_norm in {_action_normalize(a) for a in aliases}
    return False


def is_field_match(expected: str, predicted: str) -> bool:
    """
    Return True when decision + action + resource (key name + value) all match.
    For multi-ACP gold outputs (pipe-separated), a match against ANY segment counts.
    """
    if not expected.strip() or not predicted.strip():
        return False

    if expected.strip() == predicted.strip():
        return True

    # Handle pipe-separated multi-ACP gold outputs: match if ANY valid segment matches
    gold_segments = clean_gold_segments(expected)
    if not gold_segments:
        return False

    for gold in gold_segments:
        exp  = parse_acp(gold)
        pred = parse_acp(predicted)

        exp_decision  = exp.get("decision", "").lower()
        pred_decision = pred.get("decision", "").lower()
        exp_action    = exp.get("action", "").lower()
        pred_action   = pred.get("action", "").lower()

        exp_rkey,  exp_rval  = extract_resource_ref(exp.get("resource",  ""))
        pred_rkey, pred_rval = extract_resource_ref(pred.get("resource", ""))

        if (
            exp_decision == pred_decision
            and action_match(exp_action, pred_action)
            and exp_rkey == pred_rkey
            and exp_rval == pred_rval
        ):
            return True

    return False


def is_valid_segment(segment: str) -> bool:
    """Return True if this ACP segment has both a decision and an action."""
    parsed = parse_acp(segment.strip())
    return bool(parsed.get("decision", "").strip() and parsed.get("action", "").strip())


def clean_gold_segments(output: str) -> list[str]:
    """
    Split a gold output on '|' and return only the valid ACP segments.
    Drops garbage like 'decision: allow.' or 'decision: allow., reply_to_id from acp_1'.
    """
    return [s.strip() for s in output.split('|') if s.strip() and is_valid_segment(s)]


def is_valid_gold(output: str) -> bool:
    """
    Return True only if the gold output contains at least one parseable ACP
    with a non-empty decision and action field.
    Filters out empty outputs, fully unparseable rows, and rows whose only
    segments are malformed trailing fragments.
    """
    if not output or not output.strip():
        return False
    return len(clean_gold_segments(output)) > 0


# ---------------------------------------------------------------------------
# ModelGeneration
# ---------------------------------------------------------------------------

class ModelGeneration():

    def __init__(self, df: pd.DataFrame, model, tokenizer, device):
        self.inputs_gen, self.outputs_gen, self.labels_gen = [], [], []

        self.inputs, self.outputs = df['input'].to_list(), df['output'].to_list()

        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, num_beams=5):

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

            c = []   # correct predictions (field-match with gold)
            ic = []  # incorrect predictions

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

            # Skip inputs that were heavily truncated — the model sees an
            # incomplete prompt and tends to produce degenerate looping output
            if n_tokens >= 510:
                skipped += 1
                print(f"[{idx_row}] SKIP — input truncated at {n_tokens} tokens")
                continue

            ins.append(input_str)

            # ── model.generate() is the #1 hang point: wrap in try/except ──
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
                ins.pop()          # undo the append so lists stay aligned
                skipped += 1
                continue

            # ── decode is the #2 hang point: wrap each beam ──
            for beam_idx in range(num_beams):
                try:
                    result = self.tokenizer.decode(gen_outputs[beam_idx], skip_special_tokens=True)
                except Exception as e:
                    print(f"[{idx_row}] ERROR decoding beam {beam_idx}: {e} — skipping beam")
                    continue

                print(f"  beam {beam_idx}: {result[:80]}")

                # ── is_field_match is the #3 hang point (regex on bad output) ──
                try:
                    matched = is_field_match(str(gold_output), result)
                except Exception as e:
                    print(f"[{idx_row}] ERROR in is_field_match beam {beam_idx}: {e} — treating as incorrect")
                    matched = False

                if matched:
                    if result not in c:
                        c.append(result)
                else:
                    if result not in ic:
                        ic.append(result)

            corrects.append(c)
            incorrects.append(ic)

        print(f"\nGeneration complete. Skipped {skipped} rows with empty/invalid gold outputs.")
        return ins, corrects, incorrects

    def get_binary_dataset(self, num_beams=5):

        ins, corrects, incorrects = self.generate(num_beams)

        for input_str, cl, icl in zip(ins, corrects, incorrects):

            for c in cl:
                self.outputs_gen.append(c)
                self.inputs_gen.append(input_str)
                self.labels_gen.append(1)

            for ic in icl:
                self.outputs_gen.append(ic)
                self.inputs_gen.append(input_str)
                self.labels_gen.append(0)

        df = pd.DataFrame({
            'inputs': self.inputs_gen,
            'outputs': self.outputs_gen,
            'labels': self.labels_gen
        })
        return df

    def to_csv(self, name):

        df = pd.DataFrame({
            'inputs': self.inputs_gen,
            'outputs': self.outputs_gen,
            'labels': self.labels_gen
        })

        df.to_csv(name)