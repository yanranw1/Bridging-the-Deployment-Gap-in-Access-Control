import torch
import ast
import yaml
from pprint import pprint
import importlib
import pandas as pd
from torch.utils.data import Dataset, DataLoader


a,count = 0,0
failed_parse = []

class SentDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        
        self.ins = list(map(self.map_sents, df['input'].to_list()))
        self.outs = df['output'].to_list()
        
    def __len__(self):
        return len(self.ins)
    
    def map_sents(self, sent):
        sent = sent.replace(';', '').replace(":", "")
        return "Policy: " + sent + "\nEntities: {decision: " 
    
    def __getitem__(self, index):
        return {
            'input': self.ins[index],
            'output': self.outs[index]
        }
        
def data_loader(df, batch_size):
    ds = SentDataset(df)
    return DataLoader(ds, shuffle=False, batch_size=batch_size)
        
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
    
    l = ['any', 'all', 'every','the','a']
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


def make_json_sar(labels):
    global count, a, failed_parse
    ents = {'subject': [], 'action': [], 'resource': []}
    srls = {}
    policies = []
    formatted_list = []
    for l in labels:
        formatted = format_labels(l)
        formatted_list.append(formatted)
    

    for f in list(set(formatted_list)):
        a += 1
        try:
            label_json = ast.literal_eval(f)
            pp = {'subject': 'none', 'action': 'none', 'resource': 'none'}
            
            for key, val in label_json.items():
                key = key.strip()
                if key in pp:
                    nval = postprocess(key, val.strip())
                    pp[key] = nval

                    ## Added
                    if nval not in ents[key] and nval!='none':
                        ents[key].append(nval)
            policies.append(pp)

        except:
            count += 1
            # print(labels)
            failed_parse.append([labels, f])
            continue

    p = []
    
    # Sanity check
    for k,v in ents.items():
        ents[k] = list(set(v))
    
    for pol in policies:
        if pol not in p:
            p.append(pol)
            
    for pp in p:
        action = pp['action']
        if action not in srls:
            srls[action] = {'subject': [], 'resource': []}
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
        else:
            if pp['subject']!='none':
                srls[action]['subject'].append(pp['subject'])
            if pp['resource']!='none':
                srls[action]['resource'].append(pp['resource'])
        

    return p, ents, srls

            
def process_label(result, mode='sarcp'):
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
    if mode == 'sar':
        return (make_json_sar(nres))
    return(make_json(nres))
        
        
def prepare_inputs(s,l,tokenizer, device = 'cuda:4'):
    
    sent = tokenizer.encode(s, add_special_tokens=False)
    pred = tokenizer.encode(l, add_special_tokens=False)
    pair_token_ids = torch.tensor([[tokenizer.cls_token_id] + sent + [tokenizer.sep_token_id] + pred + [tokenizer.sep_token_id]])
    segment_ids = torch.tensor([[0] * (len(sent) + 2) + [1] * (len(pred) + 1)])
    attention_mask_ids = torch.tensor([[1] * (len(sent) + len(pred) + 3)])
    
    return {
        'input_ids': pair_token_ids.to(device),
        'attention_mask': attention_mask_ids.to(device),
        'token_type_ids': segment_ids.to(device)
    }
    
    
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}

def prepare_inputs_deberta(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}


def config_loader(config_file):
    with open(config_file, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    print('\n============================================ CONFIGS ============================================\n')
    pprint(data)
    print('\n=================================================================================================\n')
    
    return data


if __name__ == '__main__':
    with open('test.yaml', "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
        
    print(data)
    math = importlib.import_module(data['lib'])
    print(math.pi)
        
    
    
