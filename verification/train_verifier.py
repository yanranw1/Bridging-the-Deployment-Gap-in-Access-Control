import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
)
from sklearn.model_selection import train_test_split

import pandas as pd
import click
from verification_dataset import VerificationDataset
from dataset.utils import compute_metrics
from pathlib import Path


# ---------------------------------------------------------------------------
# Label scheme — must match component_manipulation.py
#   0  allow_deny  – flipped decision
#   1  csub        – wrong subject
#   2  cact        – wrong action
#   3  cres        – wrong resource
#   4  msub        – subject masked to 'none'
#   5  mres        – resource masked to 'none'
#   6  mrules      – rule dropped from multi-rule policy
#   7  correct     – unmodified positive example
# ---------------------------------------------------------------------------

ID2AUGS = {
    0: 'allow_deny',
    1: 'csub',
    2: 'cact',
    3: 'cres',
    4: 'msub',
    5: 'mres',
    6: 'mrules',
    7: 'correct',
}
AUGS2ID = {v: k for k, v in ID2AUGS.items()}
CORRECT_LABEL = 7   # positive class index


def train_verifier(id2augs, ds_path, batch_size=16, learning_rate=2e-5,
                   train_epochs=10, ckpt_dir='checkpoints/bart'):

    from sklearn.model_selection import StratifiedGroupKFold

    DF = pd.read_csv(ds_path)
    X, y, groups = DF.index, DF['labels'], DF['inputs']

    # First fold: 80% train / 20% holdout
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, hold_idx = next(sgkf.split(X, y, groups))
    TRAIN_DF = DF.iloc[train_idx].reset_index(drop=True)
    HOLD_DF  = DF.iloc[hold_idx].reset_index(drop=True)

    # Split holdout into val/test, again grouped+stratified
    sgkf2 = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=42)
    val_idx, test_idx = next(sgkf2.split(HOLD_DF.index, HOLD_DF['labels'], HOLD_DF['inputs']))
    VAL_DF  = HOLD_DF.iloc[val_idx].reset_index(drop=True)
    TEST_DF = HOLD_DF.iloc[test_idx].reset_index(drop=True)

    TEST_DF.to_csv('../data/verification/test_verification_dataset.csv', index=False)

    num_labels = len(id2augs)
    id2label   = {k: v for k, v in id2augs.items()}
    label2id   = {v: k for k, v in id2augs.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        'facebook/bart-large',
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

    train_ds = VerificationDataset(TRAIN_DF, tokenizer)
    val_ds   = VerificationDataset(VAL_DF,   tokenizer)
    test_ds  = VerificationDataset(TEST_DF,  tokenizer)

    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy='epoch',
        num_train_epochs=train_epochs,
        weight_decay=0.01,
        save_strategy='epoch',
        logging_strategy='epoch',
        report_to='none',
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_f1-macro',
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    testing_results = trainer.evaluate(test_ds, metric_key_prefix='test')
    print(testing_results)


@click.command()
@click.option('--dataset_path',
              default='dataset/verification_dataset.csv', required=True,
              show_default=True, help='Generated verification dataset CSV')
@click.option('--train_epochs',  default=10,   show_default=True, help='Training epochs')
@click.option('--learning_rate', default=2e-5, show_default=True, help='Learning rate')
@click.option('--batch_size',    default=8,    show_default=True, help='Batch size')
@click.option('--out_dir', default='../checkpoints/verification/checkpoint-2484',
              show_default=True, help='Checkpoint output directory')
def main(dataset_path, batch_size, learning_rate, train_epochs, out_dir):
    """Trains the access control policy verifier."""

    print('\n =========================== Training details =========================== \n')
    print(f'Dataset:       {dataset_path}')
    print(f'Num. classes:  {len(ID2AUGS)}')
    print(f'Label map:     {ID2AUGS}')
    print(f'Epochs:        {train_epochs}')
    print(f'Learning rate: {learning_rate}')
    print(f'Batch size:    {batch_size}')
    print(f'Checkpoint:    {out_dir}')
    print(' ======================================================================= \n')

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    train_verifier(
        ID2AUGS,
        ds_path=dataset_path,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_epochs=train_epochs,
        ckpt_dir=out_dir,
    )


if __name__ == '__main__':
    main()