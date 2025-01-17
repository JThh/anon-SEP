# Probe Semantic Entropy in Latent Space

## Overview

We present our probe training in the format of [notebook](./train-latent-probe.ipynb) that execute while logging and making visualizations (e.g. loss curves, performance comparisons) to enhance understanding.

## Tutorial

The approach involves retrieving the model hidden states for two token positions (TBG, SLT) on which we train linear probes to determine model semantic uncertainty or correctness.

We save model hidden states from SE generation runs (as in [model implementation](../semantic_uncertainty/uncertainty/models/huggingface_models.py)), and if you have finished SE runs using `wandb`, the model hidden states (in `validation_generations.pkl`) and uncertainty measures such as `p_true`, token `log likelihoods`, and `semantic entropy` should already be in place. And these serve as the only prerequisites of running the training notebook.

We also support saving probes (essentially a trained logistic regression model) as a pickle file to the `models` (created upon running) folder. You may run inference with the probe as you wish - it should just be a minor adaptation from the notebook that you should run the probe (SEP or Acc. Pr.) on concatenated hidden states on some particular token positions (e.g. SLT or TBG) and it will output labels (or logits) predicting how semantically certain a model is and how likely a model outputs faithful answers.

Kindly refer to our paper for terminologies and other technical details.

## Notebook Structure

This notebook is arranged in sections:

* `Imports and Downloads` helps you load wandb runs into local storage;
* `Data Preparation` section prepares the training data, encapsulates the training and evaluation codes, and contains some visualization tools;
* `Probing Acc/SE from Hidden States (IID)` section binarizes SE and carries out actual training of SEPs and Acc. Pr. in the In-Distribution setting, where we train and test on the same dataset yet on different splits;
* `Test probes trained with one dataset on others` section tests SEPs and Acc. Pr. performances in predicting model correctness on other datasets;
* The rest sections are for performance comparisons with baselines and model saving.

## Citation

```
@misc{XXXX-8,
      title={Semantic Entropy Probes: Robust and Cheap Hallucination Detection in LLMs}, 
      author={XXXX-1 and XXXX-2 and XXXX-3 and XXXX-4 and XXXX-5 and XXXX-6},
      year={2024},
      eprint={XXXX-7},
      archivePrefix={XXXX-9},
      primaryClass={cs.CL},
      url={XXXX-7}, 
}
```


