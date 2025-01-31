import json
import os
import pickle
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy
import torch
import wandb
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split


# Create train/val/test  
def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], rn=42, test_only=False, no_val=False):
    # Data splitting for sklearn linear models
    X = np.array(datasets)
    y = np.array(scores)

    if test_only:
        X_tests, y_tests = [], []
        
        for i in range(X.shape[0]):
            X_tests.append(X[i])
            y_tests.append(y)
        return (None, None, X_tests, None, None, y_tests)
    
    valid_size = val_test_splits[0]
    test_size = val_test_splits[1]

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []

    for i in range(X.shape[0]):
        # Split data into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(X[i], y, test_size=test_size, random_state=rn)
        X_tests.append(X_test)
        y_tests.append(y_test)
        if no_val:
            X_trains.append(X_train_val)
            y_trains.append(y_train_val)
            continue
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size, random_state=rn) 
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)

    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests

# Bootstrapping methods from ../semantic_entropy/uncertainty/utils/eval_utils.py
def bootstrap_func(y_true, y_score, func, rn=42):
    rng = np.random.default_rng(rn)
    y_tuple = (y_true, y_score)
    
    metric_i = func(*y_tuple)
    metric_dict = {}
    metric_dict['mean'] = metric_i
    metric_dict['bootstrap'] = compatible_bootstrap(
        func, rng)(*y_tuple)  # a bit slow to run

    return metric_dict

def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner

def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)

def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        # this function is called in the bootstrap
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func


# Train and evaluation function.
def sklearn_train_and_evaluate(model, X_train, y_train, X_valid, y_valid, silent=False):
    model.fit(X_train, y_train)
    
    # Calculate training loss and score
    train_probs = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs)

    # Calculate validation loss
    valid_preds = model.predict(X_valid)
    valid_probs = model.predict_proba(X_valid)
    valid_loss = log_loss(y_valid, valid_probs)
    val_accuracy = np.mean((valid_preds == y_valid).astype(int))
    auroc_score = roc_auc_score(y_valid, valid_probs[:,1])
    if not silent:
        print(f"Validation Accuracy: {val_accuracy:.4f}, AUROC: {auroc_score:.4f}")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")

def sklearn_evaluate_on_test(model, X_test, y_test, silent=False, bootstrap=True, rn=42):
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, test_probs)
    test_accuracy = np.mean((test_preds == y_test).astype(int))
    
    if bootstrap:
        auroc_score = bootstrap_func(y_test, test_probs[:,1], auroc, rn=rn)
        auroc_score_scalar = auroc_score['mean']
    else:
        auroc_score = auroc_score_scalar = roc_auc_score(y_test, test_probs[:, 1])
    
    if not silent:
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, AUROC: {auroc_score_scalar:.4f}")
    
    return test_loss, test_accuracy, auroc_score

def train_single_metric(D, token_type='tbg', metric='b_entropy', rn=42):
    """train and test on single metric (e.g. SE, Acc) on single dataset"""
    var_name = token_type[0] + metric[0] 
    # named as [te, se. ta, sa] for easy identification; t for tbg, s for slt (token positions)
    # e for entropy and a for accuracy (or model faithfulness)
    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
        getattr(D, f'{token_type}_dataset'), getattr(D, metric), rn=rn
    )

    accs = []
    aucs = []
    models = []
    
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        print(f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i+1}/{len(X_trains)}")
        model = LogisticRegression()
        sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test, y_test, rn=rn)
        accs.append(test_acc)
        aucs.append(test_auc)
        models.append(model)

    setattr(D, f'{var_name}_accs', accs)
    setattr(D, f'{var_name}_aucs', aucs)
    setattr(D, f'{var_name}_models', models)

# simple get-around for unpacking bootstrapping dicts
auc = lambda aucs : [ac['mean'] for ac in aucs] 
idf = lambda x : x  # identical function

# Plotting methods
def plot_metrics_ax(ax, test_metrics_list, model_names, title="", prep_func=auc, 
                    use_logarithm=False, preset_layer_indices=None, legend_outside=False):  # some simple gadgets
    """plot metrics along certain axis in a multi-axis plot (plt.subplots)"""
    if len(test_metrics_list) != len(model_names):
        raise ValueError("The length of test_metrics_list and model_names must be the same.")
    
    for test_metrics, model_name in zip(test_metrics_list, model_names):
        test_metrics = torch.tensor(prep_func(test_metrics), dtype=torch.float32)
        if use_logarithm:
            test_metrics = torch.log(test_metrics + 1e-6)
        if preset_layer_indices is not None:
            layer_indices = preset_layer_indices
        else:
            layer_indices = torch.arange(len(test_metrics)) + 1  # +1 if layer indexing starts at 1
        
        ax.plot(layer_indices, test_metrics, marker='o', linestyle='-', linewidth=2, markersize=5, label=model_name)
    
    ax.set_title(f'{title}', fontsize=14)
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel(f'Test AUROC scores', fontsize=12)
    
    tick_interval = 5  # Change this value to display more or fewer ticks
    ax.set_xticks(layer_indices[::tick_interval].tolist())
    ax.set_xticklabels(layer_indices[::tick_interval].tolist())
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if legend_outside:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    else:
        ax.legend(fontsize=12)


# Best split for SE binarization.
def best_split(entropy: torch.Tensor, label="Dx"):
    """
    Identify best split for minimizing reconstruction error via low and high SE mean estimates,
    as discussed in Section 4. Binarization of paper (ArXiv: 2406.15927)
    """
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split
    
        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])
    
        mse = np.sum((ents[low_idxs] - low_mean)**2) + np.sum((ents[high_idxs] - high_mean)**2)
        mse = np.sum(mse)
    
        split_mses.append(mse)
    
    split_mses = np.array(split_mses)
    
    plt.plot(splits, split_mses, label=label)
    return splits[np.argmin(split_mses)]

def binarize_entropy(entropy, thres=0.0):  # 0.0 means even splits for normalized entropy scores
    """Binarize entropy scores into 0s and 1s"""
    binary_entropy = torch.full_like(entropy, -1, dtype=torch.float)
    binary_entropy[entropy < thres] = 0
    binary_entropy[entropy > thres] = 1

    return binary_entropy


# select best layer range (SLT)
def decide_layer_range(Ds, metric='entropy', limit=33): # NOTE: set upperbound to be number of layers+1; e.g. for llama2-70b, it is 81.
    """
    simple logic: use ID average test AUROCs across layers to determine 
    which consecutive range of layers did the best. Do separately
    for SEP and Acc. Pr.
    """
    assert hasattr(Ds[0], 'sab_aucs') and hasattr(Ds[0], 'sa_aucs'), 'previous cells need to be run'
    if 'entropy' in metric:
        aucs = [np.array(auc(D.sab_aucs)) for D in Ds]  # test metrics for ID SEPs
    else:
        aucs = [np.array(auc(D.sa_aucs)) for D in Ds]  # test metrics for ID APs
    best_mean = -np.inf
    best_range = []
    average = lambda a,b : np.mean([np.mean(ac[a:b]) for ac in aucs])

    for i in range(limit):
        for j in range(i+1, limit):
            if j - i < 5: # must be more than 5 layers
                continue
            if average(i, j) > best_mean:
                best_mean = average(i, j)
                best_range = [i, j]

    return best_mean, best_range


# select best layer range (SLT)
def decide_layer_range_se_only(Ds, limit=33): # NOTE: set upperbound to be number of layers+1; e.g. for llama2-70b, it is 81.
    """
    simple logic: use ID average test AUROCs across layers to determine 
    which consecutive range of layers did the best. Do separately
    for SEP and Acc. Pr.
    """
    aucs = [np.array(auc(D.sb_aucs)) for D in Ds]  # test metrics for ID SEPs

    best_mean = -np.inf
    best_range = []
    average = lambda a,b : np.mean([np.mean(ac[a:b]) for ac in aucs])

    for i in range(limit):
        for j in range(i+1, limit):
            if j - i < 5: # must be more than 5 layers
                continue
            if average(i, j) > best_mean:
                best_mean = average(i, j)
                best_range = [i, j]

    return best_mean, best_range
    

def concat_Xs_and_ys(layer_range, X_trains, X_vals, X_tests, y_trains, y_vals, y_tests, 
                     no_val=False, test_only=False):
    """
    Concatenate @params{layer_range} hidden state layers on train/val/test sets.

    no_val: no validation (training set only).
    test_only: no train/validation set (test set only).
    """
    if not no_val:
        X_val_cc = np.concatenate(np.array(X_vals)[layer_range], axis=1)
        y_val_cc = y_vals[layer_range[0]]
    else:
        X_val_cc = y_val_cc = None

    if not test_only:
        X_train_cc = np.concatenate(np.array(X_trains)[layer_range], axis=1)
        y_train_cc = y_trains[layer_range[0]]
    else:
        X_train_cc = y_train_cc = None
    
    X_test_cc = np.concatenate(np.array(X_tests)[layer_range], axis=1)
    y_test_cc = y_tests[layer_range[0]]
    
    return X_train_cc, X_val_cc, X_test_cc, y_train_cc, y_val_cc, y_test_cc


def train_concat_SE(D, layer_range, rn=42):
    """train model on single dataset SE with concatenated layers"""
    for token_type in ['slt']: # optionally, ['slt', 'tbg']
        var_name = token_type[0]
        all_Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.b_entropy, test_only=True, rn=rn) # train on all data
        _, _, X_train_cc, _, _, y_train_cc = concat_Xs_and_ys(layer_range, *all_Xs_and_ys, no_val=True, test_only=True)
        model = LogisticRegression()
        model.fit(X_train_cc, y_train_cc)
        setattr(D, f'{var_name}_bmodel', model)

        print(f'{token_type.upper()} trained on {D.name} SE finished')


def train_concat_SE_with_test(D, layer_range, rn=42):
    """train model on single dataset SE with concatenated layers"""
    for token_type in ['slt']: # optionally, ['slt', 'tbg']
        var_name = token_type[0]
        all_Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.b_entropy, no_val=True, test_only=False, rn=rn) # train on all data
        X_train_cc, _, X_test_cc, y_train_cc, _, y_test_cc = concat_Xs_and_ys(layer_range, *all_Xs_and_ys, no_val=True, test_only=False)
        model = LogisticRegression()
        model.fit(X_train_cc, y_train_cc)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test_cc, y_test_cc, rn=rn) # SEP predicts error rate
        print(D.name, 'SEP ID', test_auc)
        setattr(D, f'{var_name}_bmodel', model)

        print(f'{token_type.upper()} trained on {D.name} SE finished')


def train_concat_Acc(D, layer_range, rn=42):
    """train model on single dataset Accuracy with concatenated layers"""
    for token_type in ['slt']: # optionally, ['slt', 'tbg']
        var_name = token_type[0]
        all_Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.accuracies, test_only=True, rn=rn) # train on all data
        _, _, X_train_cc, _, _, y_train_cc = concat_Xs_and_ys(layer_range, *all_Xs_and_ys, no_val=True, test_only=True)
        model = LogisticRegression()
        model.fit(X_train_cc, y_train_cc)        
        setattr(D, f'{var_name}_amodel', model)

        print(f'{token_type.upper()} trained on {D.name} Acc finished')

def train_concat_SE_Acc_test_Acc(D, layer_ranges, rn=42):
    """ID: train and test SEPs and Acc. Pr. on single dataset with concatenated layers"""
    for token_type in ['slt']: # optionally, ['slt', 'tbg']
        var_name = token_type[0]
        all_Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.b_entropy, no_val=True, rn=rn) 
        
        X_train_cc, _, _, y_train_cc, _, _ = concat_Xs_and_ys(layer_ranges[0], *all_Xs_and_ys, no_val=True)
        ab_accs = []
        ab_aucs = []
        model = LogisticRegression()
        model.fit(X_train_cc, y_train_cc)

        all_Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.accuracies, rn=rn) 
        _, _, X_test_cc, _, _, y_test_cc = concat_Xs_and_ys(layer_ranges[0], *all_Xs_and_ys, no_val=True) # fixed random seed ensures no data leakage
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test_cc, 1-y_test_cc, rn=rn) # SEP predicts error rate
        ab_accs.append(test_acc)
        ab_aucs.append(test_auc)
    
        setattr(D, f'i{var_name}b_accs', ab_accs)  # i means IDß
        setattr(D, f'i{var_name}b_aucs', ab_aucs)

        print(f'{D.name.upper()}-{token_type.upper()} trainied on SE and tested on Acc finished')
        aa_accs = []
        aa_aucs = []
        X_train_cc, _, X_test_cc, y_train_cc, _, y_test_cc = concat_Xs_and_ys(layer_ranges[1], *all_Xs_and_ys, no_val=True)
        model = LogisticRegression()
        model.fit(X_train_cc, y_train_cc)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test_cc, y_test_cc, rn=rn)
        aa_accs.append(test_acc)
        aa_aucs.append(test_auc)
    
        setattr(D, f'i{var_name}a_accs', aa_accs)
        setattr(D, f'i{var_name}a_aucs', aa_aucs)
        print(f'{D.name.upper()}-{token_type.upper()} trainied on Acc and tested on Acc finished')


# Train on one's SE and test on others' SE
def test_one_on_n_SE(D, Ds, layer_range, token_type='slt', rn=42):
    var_name = token_type[0]
    other_ids = D.other_ids
    other_names = D.other_names
    metric = 'b_entropy'
    b_model = getattr(D, f'{var_name}_bmodel')  # SE Probe

    print(f"Using probes trained on datasets {D.name.upper()}'s {token_type.upper()}-SE to predict {other_names}'s {token_type.upper()}-SE")

    ob_aucs = {}
    
    for id_ in D.other_ids:
        D_id = Ds[id_]
        print(f"Testing on {D_id.name.upper()}'s {token_type.upper()}-{metric.upper()}")
        y_metric = getattr(D_id, metric)
        
        idb_aucs = []

        # create test sets with SE labels
        all_Xs_and_ys = create_Xs_and_ys(getattr(D_id, f'{token_type}_dataset'), y_metric, test_only=True, rn=rn)
        
        # test on SE Probes
        _, _, X_test_cc, _, _, y_test_cc = concat_Xs_and_ys(layer_range, *all_Xs_and_ys, no_val=True, test_only=True)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(b_model, X_test_cc, y_test_cc, rn=rn)
        idb_aucs.append(test_auc)

        ob_aucs[D_id.name] = idb_aucs

    setattr(D, 'osbb_aucs', ob_aucs)
    print(f"Using probes trained on dataset {D.name.upper()} testing complete.")


def test_one_on_n(D, Ds, layer_range1, layer_range2, token_type='slt', rn=42):
    var_name = token_type[0]
    other_ids = D.other_ids
    other_names = D.other_names
    metric = 'accuracies'
    a_model = getattr(D, f'{var_name}_amodel')  # Acc. Probe
    b_model = getattr(D, f'{var_name}_bmodel')  # SE Probe

    print(f"Using probes trained on datasets {D.name.upper()}'s {token_type.upper()}-SE/Acc to predict {other_names}'s {token_type.upper()}-Acc")

    oa_accs = {}
    oa_aucs = {}
    ob_accs = {}
    ob_aucs = {}
    
    for id_ in D.other_ids:
        D_id = Ds[id_]
        print(f"Testing on {D_id.name.upper()}'s {token_type.upper()}-{metric.upper()}")
        if metric == 'accuracies':
            y_metric = 1 - getattr(D_id, metric)  # error rate
        else:
            y_metric = getattr(D_id, metric)
        
        ida_accs = []
        ida_aucs = []
        idb_accs = []
        idb_aucs = []

        # create test sets with accuracy labels
        all_Xs_and_ys = create_Xs_and_ys(getattr(D_id, f'{token_type}_dataset'), y_metric, test_only=True, rn=rn)

        # test on Acc. Probes
        _, _, X_test_cc, _, _, y_test_cc = concat_Xs_and_ys(layer_range2, *all_Xs_and_ys, no_val=True, test_only=True)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(a_model, X_test_cc, 1-y_test_cc, rn=rn)
        ida_accs.append(test_acc)
        ida_aucs.append(test_auc)
        
        # test on SE Probes
        _, _, X_test_cc, _, _, y_test_cc = concat_Xs_and_ys(layer_range1, *all_Xs_and_ys, no_val=True, test_only=True)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(b_model, X_test_cc, y_test_cc, rn=rn)
        idb_accs.append(test_acc)
        idb_aucs.append(test_auc)

        oa_accs[D_id.name] = ida_accs
        oa_aucs[D_id.name] = ida_aucs
        ob_accs[D_id.name] = idb_accs
        ob_aucs[D_id.name] = idb_aucs

    setattr(D, 'osa_accs', oa_accs) # o means OOD
    setattr(D, 'osa_aucs', oa_aucs)
    setattr(D, 'osb_accs', ob_accs)
    setattr(D, 'osb_aucs', ob_aucs)

    print(f"Using probes trained on dataset {D.name.upper()} testing complete.")


def merge_Xs_and_ys(Ds, other_ids, attribute='b_entropy', token_type='slt', no_val=False, rn=42):
    """Ds are the collection of dataset instances; other_ids are the other datasets"""
    print("Merging datasets")
    all_Xs_and_ys = ()
    for id_ in other_ids:
        D = Ds[id_]
        # Xs_and_ys: (X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)
        Xs_and_ys = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), getattr(D, attribute), no_val=no_val, rn=rn)
        all_Xs_and_ys += (Xs_and_ys,)
    
    # merge xs and ys by splits
    all_splits = list(zip(*all_Xs_and_ys))
    results = ()
    for splits in all_splits:
        all_layers = [[] for _ in range(len(splits[0]))]
        for split in splits:
            for i, layer in enumerate(split):
                all_layers[i].append(layer)
        results += ([np.concatenate(l, axis=0) for l in all_layers],)

    print(f"Dummy test accuracy: {max(np.mean(results[-1]), 1-np.mean(results[-1])):.4f}")
    return results    


# Train utility function
def train_on_others(D, Ds, token_type='slt', rn=42):
    var_name = token_type[0]
    other_ids = D.other_ids
    other_names = D.other_names
    
    print(f"Training on datasets {other_names}'s {token_type.upper()}-SE")

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = merge_Xs_and_ys(Ds, other_ids, 'b_entropy', token_type, rn=rn)
    ob_models = []
    
    for k, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        print(f"Training on {token_type.upper()}-SE {k+1}/{len(X_trains)} - {other_names}")
        model = LogisticRegression()
        sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
        test_loss, _, _ = sklearn_evaluate_on_test(model, X_test, y_test, rn=rn)
        ob_models.append(model)

    setattr(D, f"o{var_name}bb_models", ob_models)

    print(f"Training on datasets {other_names}'s {token_type.upper()}-Acc")

    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = merge_Xs_and_ys(Ds, other_ids, 'accuracies', token_type, rn=rn)
    oa_models = []
    
    for k, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        print(f"Training on {token_type.upper()}-Acc {k+1}/{len(X_trains)} - {other_names}")
        model = LogisticRegression()
        sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
        test_loss, _, _ = sklearn_evaluate_on_test(model, X_test, y_test, rn=rn)
        oa_models.append(model)

    setattr(D, f"o{var_name}aa_models", oa_models)


# Test
def test_n_on_one(D, token_type='slt', rn=42):
    var_name = token_type[0]
    other_ids = D.other_ids
    other_names = D.other_names
    ob_models = getattr(D, f'o{var_name}bb_models')
    oa_models = getattr(D, f'o{var_name}aa_models')
    
    print(f"Testing on {D.name}'s SE")
    
    _, _, X_tests, _, _, y_tests = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), D.b_entropy, test_only=True, rn=rn) 
    ib_accs = []
    ib_aucs = []
    
    for k, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
        # print(f"Testing on {D.name}-{token_type.upper()}-SE {k+1}/{len(X_tests)}")
        model = ob_models[k]
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test, y_test, rn=rn)
        ib_accs.append(test_acc)
        ib_aucs.append(test_auc)

    setattr(D, f'o{var_name}bb_accs', ib_accs)
    setattr(D, f'o{var_name}bb_aucs', ib_aucs)

    print(f"\nTesting on {D.name}'s Acc")
   
    _, _, X_tests, _, _, y_tests = create_Xs_and_ys(getattr(D, f'{token_type}_dataset'), 1-D.accuracies, test_only=True, rn=rn) 
    iab_accs = []
    iab_aucs = []
    iaa_accs = []
    iaa_aucs = []

    for k, (X_test, y_test) in enumerate(zip(X_tests, y_tests)):
        # print(f"Testing on {D.name}-{token_type.upper()}-Acc {k+1}/{len(X_tests)}")
        b_model = ob_models[k]
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(b_model, X_test, y_test, rn=rn)
        iab_accs.append(test_acc)
        iab_aucs.append(test_auc)
        
        a_model = oa_models[k]
        y_test = 1 - y_test  # acc predictors are not trained on error rates
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(a_model, X_test, y_test, rn=rn)
        iaa_accs.append(test_acc)
        iaa_aucs.append(test_auc)
    
    setattr(D, f'o{var_name}ba_accs', iab_accs)
    setattr(D, f'o{var_name}ba_aucs', iab_aucs)
    setattr(D, f'o{var_name}aa_accs', iaa_aucs)
    setattr(D, f'o{var_name}aa_aucs', iaa_aucs)


def save_metrics(Ds, notebook_path, file_name="x.pkl"):
    Ds_save = ()
    excl_attrs = ['tbg_dataset', 'slt_dataset', 'entropy', 'accuracies', 
                  'b_entropy', 'sa_models', 'sb_models', 'osaa_models', 'osbb_models'] 

    for D in Ds:
        D_save = {key: value for key, value in vars(D).items() if key not in excl_attrs}
        Ds_save += (D_save,)

    path = f"{notebook_path}/seed_metrics"
    full_path = os.path.expanduser(path)
    os.makedirs(full_path, exist_ok=True)

    with open(f'{full_path}/{file_name}', 'wb') as f:
        pickle.dump(Ds_save, f)
