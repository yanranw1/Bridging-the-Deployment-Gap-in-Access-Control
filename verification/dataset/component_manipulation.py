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
        
    @staticmethod
    def _pick_different(pool: list, current: str) -> str | None:
        """
        Return a random element from *pool* that differs from *current*.
        Returns None when the pool is empty or every element equals *current*
        (i.e. no valid alternative exists).

        Uses random.randrange() to avoid the off-by-one that
        ``int(random.random() * len(pool))`` has when random() == 1.0.
        """
        candidates = [v for v in pool if v != current]
        if not candidates:
            return None
        return candidates[random.randrange(len(candidates))]

    def change_rule(self, rule: dict, aug: str, _tried: frozenset = frozenset()) -> tuple:
        """
        Apply augmentation *aug* to *rule* (in-place) and return (rule, aug).

        Falls back to a different, not-yet-tried augmentation when the
        requested one cannot be applied (e.g. pool has no alternative value).
        Raises RuntimeError only when every augmentation has been exhausted.
        """
        if aug == 'allow_deny':
            rule['decision'] = 'deny' if 'allow' in rule['decision'] else 'allow'

        elif aug == 'csub':
            val = self._pick_different(self.subs, rule['subject'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['subject'] = val

        elif aug == 'cact':
            val = self._pick_different(self.acts, rule['action'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['action'] = val

        elif aug == 'cres':
            val = self._pick_different(self.ress, rule['resource'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['resource'] = val

        elif aug == 'ccond':
            val = self._pick_different(self.conds, rule['condition'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['condition'] = val

        elif aug == 'cpur':
            val = self._pick_different(self.purs, rule['purpose'])
            if val is None:
                return self._fallback(rule, aug, _tried)
            rule['purpose'] = val

        elif aug == 'msub' and rule['subject'] != 'none':
            rule['subject'] = 'none'
        elif aug == 'mres' and rule['resource'] != 'none':
            rule['resource'] = 'none'
        elif aug == 'mcond' and rule['condition'] != 'none':
            rule['condition'] = 'none'
        elif aug == 'mpur' and rule['purpose'] != 'none':
            rule['purpose'] = 'none'

        else:
            # This aug cannot be applied to the rule — try another one.
            return self._fallback(rule, aug, _tried)

        return rule, aug

    def _fallback(self, rule: dict, failed_aug: str, _tried: frozenset) -> tuple:
        """Pick a random aug from those not yet tried and retry."""
        _tried = _tried | {failed_aug}
        remaining = [a for a in self.augs if a not in _tried]
        if not remaining:
            raise RuntimeError(
                f"change_rule: no applicable augmentation found for rule {rule}. "
                f"All tried: {_tried}"
            )
        next_aug = remaining[random.randrange(len(remaining))]
        return self.change_rule(rule, next_aug, _tried)
    
    
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

                if not pols:          # skip unparseable/empty policies
                    continue

                randpol = random.randrange(len(pols))

                naug = None
                for i,rule in enumerate(pols):
                    if i == randpol:
                        rand_aug = random.randrange(len(self.augs))
                        aug = self.id2augs[rand_aug]
                        mod_rule, naug = self.change_rule(rule, aug)
                        modified.append(mod_rule)
                        self.augs[naug]+=1
                    else:
                        modified.append(rule)
                    
                sents.append(s)
                npols.append(create_out_string(modified))
                labels.append(self.augs2id[naug])
                #fix ended
                
        self.augs['mrules'] = 0

        self.augs2id['mrules'] = 10
        self.id2augs[10] = 'mrules'

        for _ in range(missing_rule_count):

            for s,p in zip(self.correct_acps['inputs'], self.correct_acps['outputs']):
                modified = []
                pols = process_label([p])
                
                if len(pols)>1:
                    rand = random.randrange(len(pols))
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