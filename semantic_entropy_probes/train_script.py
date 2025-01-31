import argparse
import json
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import *
from data_utils import *

warnings.filterwarnings("ignore")
plt.rcParams.update({'font.size': 12})


def main(seed):
    run_files = {
        'UNC_MEA': 'uncertainty_measures.pkl', 
        'VAL_GEN': 'validation_generations.pkl', # essential for SEP training
        'WAN_SUM': 'wandb-summary.json'
    } 

    rn = seed
    rng = np.random.default_rng(rn)
    ds_names = ['bioasq', 'trivia-qa', 'nq', 'squad'] 
    notebook_path = '/home/semantic_uncertainty/rebuttal/' 

    model_name = 'Llama2-7b' # replace with your model name
    n_layers = 33
    gen_form = 'short-form'
    wandb_run_ids = ['run-mb6th09c',
                     'run-x84t13ca',
                     'run-wxvq5m6a',
                     'run-9qv44yea']
    ds_paths = [f"/tmp/wandb/Llama2-7b/{run_id}" for run_id in wandb_run_ids]

    # Load all datasets
    Ds = []
    for path in ds_paths:
        Ds.append(Dataset(load_dataset(path)))

    # Set essential attributes for prettier printing
    for i, D in enumerate(Ds):
        D.name = ds_names[i]
        D.path = ds_paths[i]
        # OOD-related
        D.other_ids = [j for j in range(len(Ds)) if j != i]
        D.other_names = [ds_names[j] for j in D.other_ids]
        print(f'Dataset {D.name} fully packed')

    # Best universal split across datasets
    split_method = 'best-universal-split'
    all_entropy = torch.cat([D.entropy for D in Ds], dim=0)
    split = best_split(all_entropy, "All datasets collective")
    for D in Ds:
        D.b_entropy = binarize_entropy(D.entropy, split)
        print(f"Dummy accuracy for {D.name}: {max(torch.mean(D.b_entropy).item(), 1-torch.mean(D.b_entropy).item()):.4f}")

    # Binarized SE train/test (ID)
    for D in Ds:
        train_single_metric(D, 'slt', 'b_entropy', rn=rn)

    # Acc probes (ID)
    for D in Ds:
        train_single_metric(D, 'slt', 'accuracies', rn=rn)

    # ID SE->Acc
    for D in Ds:
        r_acc = 1 - D.accuracies

        # SLT
        _, _, X_tests, _, _, y_tests = create_Xs_and_ys(D.slt_dataset, r_acc, rn=rn)     
        sab_accs = []
        sab_aucs = []
        
        for i, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
            print(f"Testing on {D.name}-SLT {i+1}/{len(X_tests)}")
            model = D.sb_models[i]
            test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test, y_test)
            sab_accs.append(test_acc)
            sab_aucs.append(test_auc)

        D.sab_accs = sab_accs
        D.sab_aucs = sab_aucs

    # Best layer range
    emean, (e1, e2) = decide_layer_range(Ds, 'entropy', limit=n_layers)
    amean, (a1, a2) = decide_layer_range(Ds, 'acc', limit=n_layers)
    print('best layer range:', emean, (e1, e2), amean, (a1, a2))

    for D in Ds:
        D.sep_layer_range = (e1, e2)
        D.ap_layer_range = (a1, a2)

    for D in Ds:
        train_concat_SE_Acc_test_Acc(D, layer_ranges=[list(range(e1, e2)), list(range(a1, a2))], rn=rn)

    # OOD: save model trained on individual datasets for OOD tests
    for D in Ds:
        train_concat_SE(D, layer_range=list(range(e1, e2)), rn=rn)
        train_concat_Acc(D, layer_range=list(range(a1, a2)), rn=rn)

    for D in Ds:
        test_one_on_n(
            D, 
            Ds,
            layer_range1=list(range(e1, e2)), 
            layer_range2=list(range(a1, a2)), 
            rn=rn
        )

    # compute the OOD average: mean([train on B-> test on A, train on C -> test on A, train on D-> test on A])
    b_performances = defaultdict(list)
    a_performances = defaultdict(list)
    win_rate = []
    for D in Ds:
        for name in D.other_names:
            b_performances[name].append(auc(D.osb_aucs[name]))
            a_performances[name].append(auc(D.osa_aucs[name]))
            if auc(D.osb_aucs[name]) > auc(D.osa_aucs[name]):
                win_rate.append(1)
            else:
                win_rate.append(0)

    print(f"winning rate: {np.mean(win_rate)*100:.2f}%")
    for D in Ds:
        D.sep_ood_mn = np.mean(b_performances[D.name])
        D.ap_ood_mn = np.mean(a_performances[D.name])
        print(f"Average performance on {D.name}: SE Probe - {D.sep_ood_mn}, Acc Probe - {D.ap_ood_mn}")
    
    try:
        print(f"Run train on pooled others and test on one")
        for D in Ds:
            train_on_others(D, Ds, 'slt', rn=rn)
        for D in Ds:
            test_n_on_one(D, 'slt', rn=rn)
    except:
        print("Error while running pooled training on all layers")

    # log likelihood baseline
    try:
        for D in Ds:
            with open(f"{D.path}/{run_files['VAL_GEN']}", 'rb') as file:
                gens = pickle.load(file)
            liks = torch.tensor([np.mean(record['most_likely_answer']['token_log_likelihoods']) for record in gens.values()])
            # accs = torch.tensor([record['most_likely_answer']['accuracy'] for record in gens.values()])
            accs = D.accuracies  # for consistency of evaluating with GPT-4 metrics
            probs = np.exp(liks)
            D.loglik = bootstrap_func(accs, probs, auroc)
            print(f"{D.name} log-lik: {D.loglik['mean']:.4f}")
    
        # p_true and naive entropy baselines from uncertainty_measures and wandb_summary 
        # You should've enabled `--compute_p_true` in generation runs.
        for i, path in enumerate(ds_paths):
            os.chdir(path)
    
            with open(run_files['UNC_MEA'], 'rb') as file:
                measures = pickle.load(file)
    
            p_false_fixed_tuple = (measures['validation_is_false'], measures['uncertainty_measures']['p_false_fixed'])
    
            metric_i = auroc(*p_false_fixed_tuple)
            p_false_fixed_dict = {}
            p_false_fixed_dict['mean'] = metric_i
            p_false_fixed_dict['bootstrap'] = compatible_bootstrap(
                auroc, rng)(*p_false_fixed_tuple)  # a bit slow to run
    
            Ds[i].p_false_fixed = p_false_fixed_dict
    
            with open(run_files['WAN_SUM'], 'rb') as file:  # auto-collected with analyze_run.py
                boots = json.load(file)
    
            Ds[i].baseline_se = boots['uncertainty']['cluster_assignment_entropy']['AUROC']
            Ds[i].baseline_re = boots['uncertainty']['regular_entropy']['AUROC']
    
            print(f"Baselines for {Ds[i].name} computed.")
    except:
        print("Error while running baselines")

    # save models to pickle for inference
    save_metrics(Ds, notebook_path, file_name=f"{model_name}_{gen_form}_seed_{seed}.pkl")
    print('file saved.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run sep with a random seed.")
    parser.add_argument('--seed', type=int, required=True, help='Random seed for reproducibility')
    args = parser.parse_args()
    main(args.seed)
    