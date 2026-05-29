import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import click
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
from pathlib import Path
import json
import re
from utils import config_loader, data_loader
import utils
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from srl_results_sar import SRLEvalSAR

now = datetime.now()

@click.command()
@click.option('--mode', default='t2p', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber'], case_sensitive=False)
              )
@click.option('--batch_size', default=8, help='Batch size', show_default=True)
def generate(mode='collected', batch_size=8):
    
    """Batch-wise evaluation of the access control policy generation module considering only subjects, actions, and resources of the generated policies"""


    config_file = f'configs/{mode}.yaml'
    configs = config_loader(config_file)
    results_folder = configs['configs']['folder']
    result_dir = results_folder + "/sar_llama/"

    path = Path(result_dir)
    path.mkdir(parents=True, exist_ok=True)

    save_intermediate_results = configs['configs']['save_detailed_results']
    result_prefix = configs['result_prefix']
    gen_device = configs['generation']['device']

    ## Checkpoints

    ex_model_name = configs['generation']['model']
    peft_model_id = configs['generation']['checkpoint']
    dataset = configs['dataset']

    prefix = result_dir + result_prefix

    print('Loading generation model checkpoint :: ' + peft_model_id + " ...")

    base_model = AutoModelForCausalLM.from_pretrained(
        ex_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    base_model.config.pretraining_tp = 1
    base_model.config.use_cache = True

    model = PeftModel.from_pretrained(base_model, peft_model_id)
    gen_tokenizer = AutoTokenizer.from_pretrained(ex_model_name, trust_remote_code=True, padding_side='left')
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_model = model.to(gen_device)
    gen_model.eval()

    df = pd.read_csv(dataset)
    loader = data_loader(df, batch_size=batch_size)

    print(f'Total data points :: {len(df)}')

    pred_pols, true_pols, inputs = [], [], []
    predsrls, truesrls = [],[]

    for i in tqdm(loader):
        ins = i['input']
        outs = i['output']

        batch = gen_tokenizer(ins, return_tensors="pt", return_token_type_ids=False, padding=True).to(gen_device)
        
        translated = gen_model.generate(**batch,
                                        pad_token_id=gen_tokenizer.eos_token_id,
                                        do_sample=False, 
                                        # top_k=1, 
                                        max_length=512,
                                        num_return_sequences=1, 
                                        early_stopping = False,)

        results = gen_tokenizer.batch_decode(translated, skip_special_tokens=True)
        pattern = r'\{.*?\}'
        
        for k, result in enumerate(results):
            sanitized = re.findall(pattern, result)

            if len(sanitized) > 0:

                pred_pol, _, psrls = utils.process_label([sanitized[0]], mode='sar')
                true_pol, _, tsrls = utils.process_label([outs[k]], mode='sar')
                    
            else:
                pred_pol, _, psrls = utils.process_label([], mode='sar')
                true_pol, _, tsrls = utils.process_label([outs[k]], mode='sar')

            if (len(true_pol) > 0):
                inputs.append(ins[k])
                pred_pols.append(pred_pol)
                true_pols.append(true_pol)
                    
                predsrls.append(psrls)
                truesrls.append(tsrls)
            
    true_srls, pred_srls = truesrls.copy(), predsrls.copy()
    
    path = Path(result_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    if save_intermediate_results:
    
        with open(f'{prefix}_pred_srl.json', '+w') as f:
            json.dump(predsrls, f)
            
        with open(f'{prefix}_true_srl.json', '+w') as f:
            json.dump(truesrls, f)

    evaluator = SRLEvalSAR(true_srls, pred_srls)
    p,r,f1 = evaluator.get_f1()
    
    print("\n==================== Results (SAR) ====================\n")
    print(f'Mode: {mode}\n')
    print(f'Precision: {p}\nRecall: {r}\nF1: {f1}\n')
    print("========================================================\n")

    


if __name__ == '__main__':
    generate()
