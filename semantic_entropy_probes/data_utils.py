import json
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import scipy
import torch

run_files = {
    'UNC_MEA': 'uncertainty_measures.pkl', 
    'VAL_GEN': 'validation_generations.pkl', # essential for SEP training
    'WAN_SUM': 'wandb-summary.json'
} 


# Create Dataset class for easier attribute keeping
class Dataset:
    def __init__(self, values):
        self.tbg_dataset = values[0]
        self.slt_dataset = values[1]
        self.entropy = values[2]
        self.accuracies = values[3]

def load_dataset(path, n_sample=2000):
    os.chdir(path)
    
    # Read validation generated embeddings
    with open(run_files['VAL_GEN'], 'rb') as f:
        generations = pickle.load(f)
    
    # Read uncertainty measures (p-true, predictive/semantic uncertainties)
    with open(run_files['UNC_MEA'], 'rb') as g:
        measures = pickle.load(g)

    # Attribute names are hardcoded into the files
    entropy = torch.tensor(measures['uncertainty_measures']['cluster_assignment_entropy']).to(torch.float32)
    
    # accuracies = torch.tensor([record['most_likely_answer']['accuracy'] for record in generations.values()])
    accuracies = 1 - torch.tensor(measures['validation_is_false'])
    
    # hidden states for TBG (token before model generation) 
    tbg_dataset = torch.stack([record['most_likely_answer']['emb_last_tok_before_gen'] 
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)
    
    # hidden states for SLT (second last token of model generation)
    slt_dataset = torch.stack([record['most_likely_answer']['emb_tok_before_eos'] 
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)

    return (tbg_dataset[:, :n_sample, :], slt_dataset[:, :n_sample, :], entropy[:n_sample], accuracies[:n_sample])
