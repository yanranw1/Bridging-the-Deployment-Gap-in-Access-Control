import os

from pathlib import Path
import click
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import copy
import json
import re
import utils
from utils import config_loader, data_loader
import pandas as pd
from tqdm import tqdm

## Checks the confidence threshold
from evaluator import AccessEvaluator

from datetime import datetime
import ast
now = datetime.now()
        

def generate_and_verify(dataset, ex_model_name, peft_model_id, enable_verification, ver_model_name, 
                        verification_ckpt, id2augs, gen_device, ver_device):
    

    ## Loading generation model

    print('Loading generation model checkpoint :: ' + peft_model_id + " ...")

    base_model = AutoModelForCausalLM.from_pretrained(
        ex_model_name,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    base_model.config.pretraining_tp = 1
    base_model.config.use_cache = True

    gen_model = PeftModel.from_pretrained(base_model, peft_model_id).to(gen_device)
    gen_tokenizer = AutoTokenizer.from_pretrained(ex_model_name, trust_remote_code=True, padding_side='left')
    gen_tokenizer.pad_token = gen_tokenizer.eos_token

    gen_model.eval()

    ## Loading verification model
    
    if enable_verification:
        print('Loading verification model checkpoint :: ' + verification_ckpt + " ...")
        ver_tokenizer = AutoTokenizer.from_pretrained(ver_model_name)

        ver_model = AutoModelForSequenceClassification.from_pretrained(verification_ckpt).to(ver_device)
        ver_model.eval()

    ## Loading dataset

    df = pd.read_csv(dataset)
    loader = data_loader(df, batch_size=8)
    print(f'Total data points :: {len(df)}')


    pred_pols, true_pols, inputs, confs, true_confs,cats = [], [], [], [], [],[]
    predsrls, truesrls = [],[]

    ts, ps, ss = [],[],[]

    highest_confs = []

    ## Generating and verifying policies

    for i in tqdm(loader):
        torch.cuda.empty_cache()
        
        
        ins = i['input']
        outs = i['output']

        batch = gen_tokenizer(ins, return_tensors="pt", padding=True, return_token_type_ids=False).to(gen_device)

        translated = gen_model.generate(**batch,
                                    pad_token_id=gen_tokenizer.eos_token_id,
                                    do_sample=False, 
                                    max_new_tokens=512,
                                    num_return_sequences=1, 
                                    early_stopping = False,)
        
        results = gen_tokenizer.batch_decode(translated, skip_special_tokens=True)
        print(results)
        for k,result in enumerate(results):
            policies = []
            pattern = r'\{.*?\}'
            
            sanitized = re.findall(pattern, result)
            print(sanitized)

            if len(sanitized) > 0:
                p = sanitized[0]
                pred_pol,_ = utils.process_label([p])
                print(pred_pol)
                
                if len(pred_pol) > 0:
                    pp = utils.create_out_string(pred_pol)
                    if enable_verification:
                        with torch.no_grad():
                            ver_inp = utils.prepare_inputs_bart(ins[k], pp, ver_tokenizer, ver_device)
                            pred = ver_model(**ver_inp).logits

                            probs = torch.softmax(pred, dim=1)
                        prob_correct = probs[0][-1]
                        highest_prob = probs.max().item()
                        highest_prob_cat = id2augs[probs.argmax(axis=-1).item()]
                    else:
                        prob_correct = 1
                        highest_prob = 1
                        highest_prob_cat = 'correct'

                    policies.append([pp, prob_correct, [highest_prob_cat, highest_prob]])

            else:
                pred_pol,_ = utils.process_label([])
                pp = '{}'
                policies.append([pp, 0, ['mrules', 1]])

            if len(policies) > 0:

                pol_sorted = policies

                highest_conf_pol = pol_sorted[0][0]
                
                
                if highest_conf_pol != '{}':
                    pred_pol, srlpred = utils.process_label([highest_conf_pol])
                    highest_conf = pol_sorted[0][1]
                    cat = pol_sorted[0][2]
                else:
                    pred_pol, srlpred = utils.process_label([])
                    highest_conf = 0
                    cat = ['mrules',1]
                true_pol, srltrue = utils.process_label([outs[k]])

                
                if enable_verification:
                    with torch.no_grad():
                        true_inp = utils.prepare_inputs_bart(ins[k], outs[k], ver_tokenizer, ver_device)
                        true_prob = torch.softmax(ver_model(**true_inp).logits, dim=1)[0][1]
                
                else:
                    true_prob = 1
                
                cats.append(cat)
                highest_confs.append(highest_conf)

                inputs.append(ins[k])
                pred_pols.append(pred_pol)
                true_pols.append(true_pol)
                
                predsrls.append(srlpred)
                truesrls.append(srltrue)
                
                confs.append(highest_conf)
                
                true_confs.append(true_prob)
                    # del true_inp
                
    true_srls, pred_srls = copy.deepcopy(truesrls), copy.deepcopy(predsrls)
    e = AccessEvaluator(true_pols, pred_pols, true_srls, pred_srls, inputs, confs, true_confs, cats=cats)
    print(f"Non-empty predictions: {sum(1 for p in pred_pols if p)}")
    print(f"Sample result (raw): {results[0]}")  # first raw model output 
    results,incorrect_preds = e.evaluate()
    
    return results,incorrect_preds,truesrls, predsrls

@click.command()
@click.option('--mode', default='t2p', 
              help='Mode of training (document-fold you want to evaluate the trained model on)',
              show_default=True,
              required=True,
              type=click.Choice(['t2p', 'acre', 'ibm', 'collected', 'cyber','overall',"customized"], case_sensitive=False))
def main(mode):
    
    """Batch-wise evaluation of the access control policy generation module considering subjects, actions, resources, purposes, and conditions, as well as individual ACR decisions"""
    
    config_file = f'configs/{mode}.yaml'
    configs = config_loader(config_file)
    results_folder = configs['configs']['folder']
    
    results_dir = results_folder + "/sarcp_llama/"

    path = Path(results_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    save_intermediate_results = configs['configs']['save_detailed_results']
    result_prefix = configs['result_prefix']
    id2augs = {0: 'allow_deny',
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


    gen_device = configs['generation']['device']

    ## Checkpoints

    ex_model_name = configs['generation']['model']
    peft_model_id = configs['generation']['checkpoint']

    enable_verification = configs['verification']['enable']
    
    ver_model_name = None
    verification_ckpt = None
    ver_device = None
    if enable_verification:
        ver_model_name = configs['verification']['model']
        verification_ckpt = configs['verification']['checkpoint']  # length 512
        ver_device = configs['verification']['device']

    dataset = configs['dataset']
    
    prefix = results_dir + result_prefix

    results, _, truesrls, predsrls = generate_and_verify(dataset, ex_model_name, peft_model_id, enable_verification, ver_model_name, verification_ckpt,
                        id2augs, gen_device, ver_device)
    
    if save_intermediate_results:
        
        with open(f'{prefix}_true_srl.json', 'w') as f:
            json.dump(truesrls, f)
                
        with open(f'{prefix}_pred_srl.json', 'w') as f:
            json.dump(predsrls, f)

    
    print("\n\n ========================= Results (SARCP) ================================= \n")
    print(f'Mode: {result_prefix}\n')
    print(f"Precision: {results['srl']['precision']}\nRecall: {results['srl']['recall']}\nF1: {results['srl']['f1']}\n")
    print(f"Component-wise results: {results['srl']['components']}\n")
    
    print("\n ========================= Results (ACR Generation) ========================= \n")
    
    print(f"Precision: {results['ent_type']['precision']}\nRecall: {results['ent_type']['recall']}\nF1: {results['ent_type']['f1-score']}\n")
    print(" ============================================================================ \n\n")

                
if __name__ == '__main__':
    main()
