from utils import process_label, is_equal, create_out_string
import pandas as pd
from tqdm import tqdm
import re

class ModelGeneration():
    
    def __init__(self, df: pd.DataFrame, model, tokenizer, device):
        self.inputs_gen, self.outputs_gen, self.labels_gen = [],[],[]
        
        self.inputs, self.outputs = df['input'].to_list(), df['output'].to_list()
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
    def generate(self, num_beams=5):
        
        ins, corrects, incorrects = [],[],[]
        
        for i,o in tqdm(zip(self.inputs, self.outputs)):
            c = []
            ic = []
            ins.append(i)
            labels = process_label([o])
            c.append(labels)
            
            input = "Policy: "+ i + "\nEntities: {decision: "
            device = "cuda:0"

            print(input)

            inputs = self.tokenizer(input, return_tensors="pt", return_token_type_ids=False).to(device)
            outputs = self.model.generate(**inputs, 
                                    pad_token_id=self.tokenizer.eos_token_id,
                                    do_sample=True,
                                    top_k=10,
                                    top_p=0.95,
                                    temperature=0.8,
                                    max_length=512, 
                                    num_return_sequences=num_beams, 
                                    early_stopping = True,
                                    #  force_words_ids = force_words_ids
                                    )
            for i in range(num_beams):
                
                result = self.tokenizer.decode(outputs[i], skip_special_tokens=True)

                pattern = r'\{.*?\}'
                sanitized = re.findall(pattern, result)
                if len(sanitized)>0:
                    preds = process_label([sanitized[0]])
                    if (is_equal(preds, labels)):
                        if preds not in c:
                            c.append(preds)
                    elif is_equal(preds, labels) == False and preds not in ic:
                        ic.append(preds)
                        
            corrects.append(c)
            incorrects.append(ic)
            
        return ins, corrects, incorrects
    
    
    def get_binary_dataset(self, num_beams = 5):
        
        ins, corrects, incorrects = self.generate(num_beams)
        for i,cl,icl in zip(ins, corrects, incorrects):
            
            for c in cl:
                cstr = create_out_string(c)
                self.outputs_gen.append(cstr)
                self.inputs_gen.append(i)
                self.labels_gen.append(1)
                
            for ic in icl:
                icstr = create_out_string(ic)
                self.outputs_gen.append(icstr)
                self.inputs_gen.append(i)
                self.labels_gen.append(0)
                
        
        df = pd.DataFrame({
            'inputs': self.inputs_gen,
            'outputs': self.outputs_gen,
            'labels': self.labels_gen
        })        
        return df
             
            
    
    def to_csv(self, name):
        
        df = pd.DataFrame({
            'inputs': self.inputs_gen,
            'outputs': self.outputs_gen,
            'labels': self.labels_gen
        })
        
        df.to_csv(name)