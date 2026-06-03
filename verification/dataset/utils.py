import ast
import numpy as np
from evaluate import load
from transformers import EvalPrediction

def compute_metrics(eval_pred: EvalPrediction):
   load_accuracy = load("accuracy")
   load_f1 = load("f1")

   print("debug2",logits.shape)
   
   logits = eval_pred.predictions[0]
   
   labels = eval_pred.label_ids
   predictions = np.argmax(logits, axis=-1)
   
   accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
   f1_micro = load_f1.compute(predictions=predictions, references=labels, average = 'micro')["f1"]
   f1_macro = load_f1.compute(predictions=predictions, references=labels, average = 'macro')["f1"]
   f1_weighted = load_f1.compute(predictions=predictions, references=labels, average = 'weighted')["f1"]
   return {"accuracy": accuracy, "f1-micro": f1_micro, "f1-macro": f1_macro, "f1-weighted": f1_weighted}

a,count = 0,0
failed_parse = []

def create_out_string(inp):
    if not inp:
        return "{}"
    s = "{"
    for e in inp:
        for k,v in e.items():
            s+=f"{k}: {v}; "
            
        s = s[:-2] + " | "
        
    return s[:-3] + "}"

import re as _re

# Canonical ACP fields, in no particular order. We locate each "field:" marker
# inside a rule string and take everything up to the next marker as its value.
_FIELDS = ['decision', 'subject', 'action', 'resource', 'condition', 'purpose']
_FIELD_MARKER_RE = _re.compile(
    r'\b(' + '|'.join(_FIELDS) + r')\s*:\s*',
    _re.IGNORECASE,
)

def parse_rule(rule: str):
    """Parse a single ACP rule string into a field->value dict.

    Splits on the known field markers (decision:, subject:, action:, resource:,
    condition:, purpose:) wherever they appear, so commas/colons/'=' inside a
    value (e.g. a resource field) are preserved instead of breaking parsing.
    Returns None if no recognised fields are found.
    """
    # Strip outer braces and surrounding whitespace.
    text = rule.strip()
    if text.startswith('{'):
        text = text[1:]
    if text.endswith('}'):
        text = text[:-1]
    text = text.strip()
    if not text:
        return None

    markers = list(_FIELD_MARKER_RE.finditer(text))
    if not markers:
        return None

    pp = {'decision': 'allow', 'subject': 'none', 'action': 'none',
          'resource': 'none', 'condition': 'none', 'purpose': 'none'}

    for idx, m in enumerate(markers):
        key = m.group(1).lower()
        start = m.end()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        val = text[start:end].strip()
        # Trim a trailing field separator left between fields (";" or ",").
        val = val.rstrip().rstrip(';').rstrip(',').strip()
        if key in pp:
            pp[key] = val.lower()
    return pp


def format_labels(label: str):
    # Retained for backward compatibility; parsing now happens in parse_rule.
    return label


def make_json(labels):
    global count, a, failed_parse
    policies = []

    for f in list(set(labels)):
        a += 1
        pp = parse_rule(f)
        if pp is None:
            count += 1
            failed_parse.append([labels, f])
            continue
        policies.append(pp)

    p = []
    for pol in policies:
        if pol not in p:
            p.append(pol)

    return p
            
            
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


def longest_common_substring(str1, str2):
        # Initialize a matrix to store the lengths of common substrings
        # dp[i][j] will store the length of the longest common substring ending at str1[i-1] and str2[j-1]
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]
        
        # Variables to store the length of the longest common substring and its ending index
    longest_substring_length = 0
    longest_substring_end_index = 0
        
        # Fill the matrix
    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest_substring_length:
                    longest_substring_length = dp[i][j]
                    longest_substring_end_index = i - 1
            else:
                dp[i][j] = 0
        
        # Extract the longest common substring
    longest_substring_start_index = longest_substring_end_index - longest_substring_length + 1
    longest_substring = str1[longest_substring_start_index:longest_substring_end_index + 1]
        
    return longest_substring, longest_substring_start_index, longest_substring_end_index

def do_overlap(x, y, thresh = 1):
        
    overlap, start, end = longest_common_substring(x.lower(), y.lower())
    max_len =  len(x) #max(len(x), len(y))
        
    seg_len = end - start +1
        
    if (seg_len/max_len >= thresh):
        return True
        
    return False


def is_equal(preds, labels):
    pcopy = preds.copy()
    lcopy = labels.copy()
    
    found = []
    
    if len(preds) != len(labels):
        return False
    else:
        for pred in preds:
            d = pred['decision']
            s = pred['subject']
            a = pred['action']
            r = pred['resource']
            p = pred['purpose']
            c = pred['condition']
            
            if pred in labels and pred not in found:
                found.append(pred)
                lcopy.remove(pred)
                pcopy.remove(pred)
                
        for pred in preds:
            if pred not in found:
                d = pred['decision']
                s = pred['subject']
                a = pred['action']
                r = pred['resource']
                p = pred['purpose']
                c = pred['condition']
                
                for l in labels:
                    if l in found:
                        continue
                    dl = l['decision']
                    sl = l['subject']
                    al = l['action']
                    rl = l['resource']
                    pl = l['purpose']
                    cl = l['condition']
                    
                    if do_overlap(dl, d) and do_overlap(sl, s) and do_overlap(al, a) and do_overlap(rl, r, 0.8) and do_overlap(pl, p, 0.2) and do_overlap(cl, c, 0.2):
                        found.append(l)
                        lcopy.remove(l)
                        pcopy.remove(pred)
                        
                        break

        if len(pcopy) == len(lcopy) == 0:
            return True
        else:
            return False
        
def prepare_inputs_bart(s,l,tokenizer, device = 'cuda:0'):
    
    tokens = tokenizer(s,l,return_tensors='pt')
    
    return {k:v.to(device) for k,v in tokens.items()}