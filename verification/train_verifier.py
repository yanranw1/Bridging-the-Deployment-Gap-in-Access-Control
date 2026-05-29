import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from transformers import (
    AutoTokenizer,
    Trainer,
    AutoTokenizer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
from sklearn.model_selection import train_test_split

import pandas as pd
import click
from verification_dataset import VerificationDataset
from dataset.utils import compute_metrics
from pathlib import Path


def train_verifier(id2augs, ds_path='../datasets/verification/combined_verification_acps_missing_rules_phi_errorids_train_250.csv'
                   , batch_size = 16, learning_rate = 2e-5, train_epochs =10, ckpt_dir = 'checkpoints/bart'):

    DF = pd.read_csv(ds_path)
    TRAIN_DF, VAL_DF = train_test_split(DF, test_size=0.2, random_state=42)
    VAL_DF, TEST_DF = train_test_split(VAL_DF, test_size=0.5, random_state=42)
    
    TEST_DF.to_csv('../data/verification/test_verification_dataset.csv')

    model = AutoModelForSequenceClassification.from_pretrained('facebook/bart-large', num_labels = len(id2augs))
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')

        
    train_ds = VerificationDataset(TRAIN_DF, tokenizer)
    val_ds = VerificationDataset(VAL_DF, tokenizer)
    test_ds = VerificationDataset(TEST_DF, tokenizer)
    
    path = Path(ckpt_dir)
    path.mkdir(parents=True, exist_ok=True)

    
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_steps= 100,
        evaluation_strategy='steps',
        num_train_epochs=train_epochs,
        weight_decay=0.01,
        save_strategy="steps",
        save_steps=100,
        logging_steps=100,
        report_to='none',
        load_best_model_at_end=True,
        save_total_limit=1,
        metric_for_best_model='eval_f1-macro',
        greater_is_better=True
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
              help='Location of the generated verification dataset',
              show_default=True,
              default='../data/verification/verification_dataset.csv',
              required=True)
@click.option('--train_epochs', default=10, help='Number of epochs to train',show_default=True)
@click.option('--learning_rate', default=2e-5, help='Learning rate',show_default=True)
@click.option('--batch_size', default=8, help='Batch size',show_default=True)
@click.option('--out_dir', default='../checkpoints/verification/', help='Output directory',show_default=True)
def main(dataset_path, batch_size = 16, learning_rate = 2e-5, train_epochs = 10, 
                   out_dir = 'checkpoints/bart'):
    
    """Trains the access control policy verifier"""
    
    ID2AUGS = {0: 'allow_deny',
                1: 'csub',
                2: 'cact',
                3: 'cres',
                4: 'ccond',
                5: 'cpur',
                6: 'msub',
                7: 'mres',
                8: 'mcond',
                9: 'mpur',
                10: 'mrules',
                11: 'correct'}
    
    print('\n =========================== Trining details =========================== \n')
    print(f'Dataset: {dataset_path}\nNum. of classes: {len(ID2AUGS)}\nNum. of epochs: {train_epochs}\nLearning rate: {learning_rate}\nBatch size: {batch_size}\nCheckpoint dir.: {out_dir}\n')
    print(' ======================================================================= \n')

    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    train_verifier(ID2AUGS, ds_path=dataset_path, batch_size = batch_size, learning_rate = learning_rate, train_epochs=train_epochs, ckpt_dir = out_dir)
    

if __name__ == '__main__':
    main()