from utils import process_label, create_out_string
import random
import pandas as pd

class ComponentManipulation():
    
    def __init__(self, df) -> None:
        
        
        self.augs = {'allow_deny':0, 'csub':0, 'cact': 0, 'cres': 0, 'ccond': 0, 'cpur':0, 'msub':0, 'mres': 0, 'mcond': 0, 'mpur': 0}
        self.augs2id = {'allow_deny': 0, 'csub': 1, 'cact': 2, 'cres': 3, 'ccond': 4, 'cpur': 5, 'msub': 6, 'mres': 7, 'mcond': 8, 'mpur': 9}
        self.id2augs = {0: 'allow_deny', 1: 'csub', 2: 'cact', 3: 'cres', 4: 'ccond', 5: 'cpur', 6: 'msub', 7: 'mres', 8: 'mcond', 9: 'mpur'}
        self.correct_acps = df[df['labels']==1]
        self.subs, self.acts, self.ress, self.purs, self.conds = self.collect_components()
        # print(len(self.subs), len(self.acts), len(self.ress), len(self.purs), len(self.conds))
    
    def collect_components(self):
        
        subs, acts, ress, purs, conds = [],[],[],[],[]
        outs = self.correct_acps['outputs'].to_list()
        for policy in outs:
            pol = process_label([policy])
            for rule in pol:
                s = rule['subject']
                a = rule['action']
                r = rule['resource']
                p = rule['purpose']
                c = rule['condition']
                
                if s not in subs and s!='none':
                    subs.append(s)
                if a not in acts and a!='none':
                    acts.append(a)
                if r not in ress and r!='none':
                    ress.append(r)
                if p not in purs and p!='none':
                    purs.append(p)
                if c not in conds and c!='none':
                    conds.append(c) 
                    
        return subs, acts, ress, purs, conds
        
    def change_rule(self,rule, aug):
    
        if aug == 'allow_deny':
            if 'allow' in rule['decision']:
                rule['decision'] = 'deny'
            else:
                rule['decision'] = 'allow'
                
        elif aug == 'csub':
            rand = int(random.random() * len(self.subs))
            while self.subs[rand]==rule['subject']:
                rand = int(random.random() * len(self.subs))
                
            rule['subject'] = self.subs[rand]
            
        elif aug == 'cact':
            rand = int(random.random() * len(self.acts))
            while self.acts[rand]==rule['action']:
                rand = int(random.random() * len(self.acts))
                
            rule['action'] = self.acts[rand]
            
        elif aug == 'cres':
            rand = int(random.random() * len(self.ress))
            while self.ress[rand]==rule['resource']:
                rand = int(random.random() * len(self.ress))
                
            rule['resource'] = self.ress[rand]
            
        elif aug == 'ccond' and len(self.conds)>1:
            rand = int(random.random() * len(self.conds))
            # print(len(self.conds), rand)
            while self.conds[rand]==rule['condition']:
                # print(rand)
                rand = int(random.random() * len(self.conds))
                
            rule['condition'] = self.conds[rand]
            
        elif aug == 'cpur' and len(self.purs)>1:
            rand = int(random.random() * len(self.purs))
            # print(len(self.purs), rand)
            while self.purs[rand]==rule['purpose']:
                # print(rand)
                rand = int(random.random() * len(self.purs))
                
            rule['purpose'] = self.purs[rand]
            
            
        
        elif aug == 'msub' and rule['subject'] != 'none':
            rule['subject'] = 'none'
        elif aug == 'mres' and rule['resource'] != 'none':
            rule['resource'] = 'none'
        elif aug == 'mcond' and rule['condition'] != 'none':
            rule['condition'] = 'none'
        elif aug == 'mpur' and rule['purpose'] != 'none':
            rule['purpose'] = 'none'
            
        else:
            
            rand_aug = int(random.random() * len(self.augs))
            augn = list(self.augs.keys())[rand_aug]
            
            return self.change_rule(rule, augn)
        # augs[aug]+=1
        return rule, aug
    
    
    def get_summary(self, df: pd.DataFrame):
        grouped = df.groupby('labels')['labels'].count()
        return grouped
    
    
    def augment(self, num_times=5, missing_rule_count = 4, print_summary=True):
        
        sents = []
        npols = []
        labels = []
        
        for _ in range(num_times):
            for s,p in zip(self.correct_acps['inputs'], self.correct_acps['outputs']):
                modified = []
                pols = process_label([p])
                randpol = int(random.random() * len(pols))
                for i,rule in enumerate(pols):
                    if i == randpol:
                        rand_aug = int(random.random() * len(self.augs))
                        aug = self.id2augs[rand_aug]
                        mod_rule, naug = self.change_rule(rule, aug)
                        modified.append(mod_rule)
                        labels.append(self.augs2id[naug])
                        self.augs[naug]+=1
                    else:
                        modified.append(rule)
                    
                sents.append(s)
                npols.append(create_out_string(modified))
                
        self.augs['mrules'] = 0

        self.augs2id['mrules'] = 10
        self.id2augs[10] = 'mrules'

        for _ in range(missing_rule_count):

            for s,p in zip(self.correct_acps['inputs'], self.correct_acps['outputs']):
                modified = []
                pols = process_label([p])
                
                if len(pols)>1:
                    rand = int(random.random() * len(pols))
                    pols.pop(rand)
                    
                    sents.append(s)
                    npols.append(create_out_string(pols))
                    labels.append(10)

                    self.augs['mrules']+=1
                    
        csents, cpols = self.correct_acps['inputs'].to_list(), self.correct_acps['outputs'].to_list()
        clabels = [11]*len(csents)

        sents.extend(csents)
        npols.extend(cpols)
        labels.extend(clabels)
        
        df = pd.DataFrame({
            'inputs': sents,
            'outputs': npols,
            'labels': labels
        })
        
        if print_summary:
            print(self.get_summary(df))
        
        return df
        
        
    