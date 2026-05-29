import pandas as pd
from sklearn.model_selection import train_test_split
import argparse
import os

parser = argparse.ArgumentParser(description="Convert JSON dataset and create train/test splits")
parser.add_argument("input_json", help="Path to input JSON file")

args = parser.parse_args()

# Load JSON
df = pd.read_json(args.input_json)

# Transform columns
new_df = pd.DataFrame({
    "input": df["NL"],
    "acp": 1,
    "output": df["ACP"]
})

# Train/test split
train_df, test_df = train_test_split(
    new_df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# Create output directory
output_dir = "data/customized"
os.makedirs(output_dir, exist_ok=True)

# Save CSVs
train_path = os.path.join(output_dir, "train.csv")
test_path = os.path.join(output_dir, "test.csv")

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"Train set saved to: {train_path}")
print(f"Test set saved to: {test_path}")

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")