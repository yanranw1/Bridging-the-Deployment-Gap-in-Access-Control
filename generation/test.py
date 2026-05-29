import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "meta-llama/Meta-Llama-3-8B"
LORA_PATH = "../checkpoints/llama3/ibm/checkpoint-40"
CSV_PATH = "../data.csv"

device = "cuda"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_PATH)

model.eval()

# Load CSV
df = pd.read_csv(CSV_PATH)

predictions = []

for idx, row in df.iterrows():

    policy = row["input"]
    expected = row["output"]

    prompt = f"Policy: {policy}\nEntities:"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    # Extract only generated part
    prediction = decoded.split("Entities:")[-1].strip()

    predictions.append(prediction)

    print("\n===================================")
    print(f"Example {idx+1}")
    print("Policy:")
    print(policy)
    print("\nExpected:")
    print(expected)
    print("\nPredicted:")
    print(prediction)

# Save outputs
df["prediction"] = predictions

df.to_csv("evaluation_results.csv", index=False)

print("\nSaved results to evaluation_results.csv")
