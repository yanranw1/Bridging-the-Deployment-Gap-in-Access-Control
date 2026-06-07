import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import click
from transformers import EvalPrediction
import numpy as np
from metrics import compute_metrics
import pandas as pd
from classification_dataset import ACPDataset
from scipy.special import softmax  # Used to get confidence probabilities
from transformers import BertForSequenceClassification, BertTokenizerFast
from torch.utils.data import DataLoader

def evaluate(loader, model, device):
    
    all_preds = None
    all_labels = None

    for i,batch in enumerate(loader):
        input = {k: v.to(device) for k,v in batch.items()}

        out = model(**input)
        
        logits = out.logits.cpu().detach().numpy()
        labels = input['labels'].cpu().detach().numpy()
        
        all_preds = logits if all_preds is None else np.concatenate((all_preds, logits))
        all_labels = labels if all_labels is None else np.concatenate((all_labels, labels))
        
    eval_pred = EvalPrediction(predictions = all_preds, label_ids=all_labels)
    return compute_metrics(eval_pred),all_preds

@click.command()
@click.option('--mode', default='email', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber', 'overall',"email"], case_sensitive=False)
              )
@click.option('--batch_size', default=16, help='Batch size', show_default=True)
@click.option('--device', default='cuda:0', help='GPU/CPU', show_default=True)
def main(mode, batch_size = 16, device = 'cuda:0'):
    
    """ Evaluates the NLACP identification module."""
    
    NUM_CLASSES = 2
    
    if mode=='overall':
        test_path = f'../data/overall/test.csv'
    elif mode=='email':
        test_path = "/home/ubuntu/agentv-main/email_agent/dataset/combined_with0_test.csv"
    else:
        test_path = f'../data/document_folds/{mode}.csv'
        
    checkpoint = f'../checkpoints/identification/{mode}'

    model = BertForSequenceClassification.from_pretrained(checkpoint, num_labels=NUM_CLASSES).to(device)
    tokenizer = BertTokenizerFast.from_pretrained(checkpoint)

    
    test_df = pd.read_csv(test_path)
    test_ds = ACPDataset(test_df, tokenizer)
    test_dataloader = DataLoader(test_ds, num_workers=1, batch_size=batch_size)
    
    print('\n =============================== Evaluating ================================ \n')
    
    
    results, all_preds= evaluate(test_dataloader, model, device)
    
    print(f'Test path: {test_path}\n')
    print(results)
    print('\n =========================== Saving Predictions ============================ \n')
    
    # 1. Get the highest scoring class index (0 or 1) for each row
    pred_labels = np.argmax(all_preds, axis=1)
    test_df['predicted_label'] = pred_labels
    
    # 2. Optional: Extract the confidence score (probability) for the predicted class
    probs = softmax(all_preds, axis=1)
    test_df['confidence_score'] = np.max(probs, axis=1)

    # 3. Choose how you want to save the data:
    identified_df = test_df[test_df['predicted_label'] == 1]
    
    # Option A: Save ALL entries with their new prediction columns
    output_all_path = test_path.replace('.csv', '_identified_only.csv')
    identified_df.to_csv(output_all_path, index=False)
    print(f"All predictions saved to: {output_all_path}")
    print('\n =========================================================================== \n')
    
    
if __name__ == '__main__':
    main()
    
    