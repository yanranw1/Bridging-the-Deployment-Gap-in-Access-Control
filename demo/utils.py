import torch
import ast
import yaml
from pprint import pprint
import re
import numpy as np


a,count = 0,0
failed_parse = []
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

def create_out_string(inp):
    s = "{"
    # print(inp)
    for e in inp:
        # print(e)
        for k, v in e.items():
            s += f"{k}: {v}; "

        s = s[:-2] + " | "

    return s[:-3] + "}"

def format_labels(label: str):
    nlabel =  label.lower().replace('"', '').replace(';',',').replace('decision: ', "\'decision\': '").replace('subject: ', "\'subject\': '").replace('action: ', "\'action\': '").replace('resource: ', "\'resource\': '").replace('condition: ', "\'condition\': '").replace('purpose: ', "\'purpose\': '").replace(",", "',").replace('}', "'}").replace("'s ", "\\'s ")
    i = nlabel.find('decision')
    return ("{'" + nlabel[i:])

def postprocess(key, val):
    
    l = ['any', 'all', 'every','the']
    conds = ['when', 'if', 'if,']
    purs = ['for', 'to']
    
    val = val.replace('_',',').replace('-', '').replace("'", "").replace('’', '').replace("”", "").replace("“","")
    
    if key == 'subject' or key == 'resource':
        for k in l:
            if val.split(' ')[0] == k:
                val = ' '.join(val.split(' ')[1:])
            
    if key=='purpose' and val.split(' ')[0] in purs:
        val = ' '.join(val.split(' ')[1:])
        
    if key=='condition' and val.split(' ')[0] in conds:
        val = ' '.join(val.split(' ')[1:])
    
    if val[-1] == 's' and val[-2] != 's':
        val = val[:-1]
        
    return ' '.join(val.split())


def make_json(labels):
    global count, a, failed_parse
    policies = []
    formatted_list = []
    srls = {}
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)
    
    for f in list(set(formatted_list)):
        a+=1
        try:
            label_json = ast.literal_eval(f)
            pp = {'decision': 'allow', 'subject': 'none', 'action': 'none', 'resource': 'none', 'condition': 'none', 'purpose': 'none'}
            for key,val in label_json.items():
                key = key.strip()
                if key in pp:
                    pp[key] = postprocess(key, val.strip())
            policies.append(pp)
            
            action = pp['action']
            if action not in srls:
                srls[action] = {'subject': [], 'resource': [], 'purpose': [], 'condition': []}
                if pp['subject']!='none':
                    srls[action]['subject'].append(pp['subject'])
                if pp['resource']!='none':
                    srls[action]['resource'].append(pp['resource'])
                if pp['purpose']!='none':
                    srls[action]['purpose'].append(pp['purpose'])
                if pp['condition']!='none':
                    srls[action]['condition'].append(pp['condition'])    
            else:
                if pp['subject']!='none':
                    srls[action]['subject'].append(pp['subject'])
                if pp['resource']!='none':
                    srls[action]['resource'].append(pp['resource'])
                if pp['purpose']!='none':
                    srls[action]['purpose'].append(pp['purpose'])
                if pp['condition']!='none':
                    srls[action]['condition'].append(pp['condition'])
        
        except:
            count+=1
            failed_parse.append([labels, f])
            continue
        
    p = []
    for pol in policies:
        if pol not in p:
            p.append(pol)
        
    return p, srls
            
            
def process_label(result):
    res = []
    if (len(result) > 0):
        for p in result:
            ind = p.split(" | ")
            if (len(ind) == 1):
                res.append(ind[0])
            else:
                for i in range(len(ind)):
                    if (i==0 and ind[i][-1]!="}"):
                        res.append(ind[i]+"}")
                    elif (i == len(ind)-1 and ind[i][0]!="{"):
                        res.append("{" + ind[i])
                    else:
                        res.append("{" + ind[i] + "}")
    nres = list(set(res))
    return(make_json(nres))

    
    
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}

        
def generate_policy(input, id_tokenizer, id_model, gen_tokenizer, gen_model, device):

    tokens = id_tokenizer(input, return_token_type_ids = False, return_tensors="pt").to(device)

    out = id_model(**tokens)
    res = torch.argmax(out.logits, dim=1)

    if res.item() == 1:

        input_text = "Policy: " + input + "\nEntities: {decision: "
        batch = gen_tokenizer(input_text, return_tensors="pt", return_token_type_ids=False).to(device)
        translated = gen_model.generate(**batch,
                                    pad_token_id=gen_tokenizer.eos_token_id,
                                    do_sample=False,
                                    max_length=512,
                                    num_return_sequences=1,
                                    early_stopping = False,)
        result = gen_tokenizer.decode(translated[0], skip_special_tokens=True)
        pattern = r'\{.*?\}'
        sanitized = re.findall(pattern, result)
        return sanitized[0],1
    else:
        return "Not an ACP",0
    
def verify_policy(nlacp, policy, ver_tokenizer, ver_model, device):

    pred_pol,_ = process_label([policy])
    pp = create_out_string(pred_pol)
    ver_inp = prepare_inputs_bart(nlacp, pp, ver_tokenizer, device)
    with torch.no_grad():
        preds = ver_model(**ver_inp).logits
    probs = torch.softmax(preds, dim=1)

    mprob = probs.max().item()
    pred = probs.argmax(axis=-1).item()
    highest_prob_cat = ID2AUGS[pred]

    return highest_prob_cat, pred, mprob
