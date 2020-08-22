# BET: Transformer training scripts

This repository have scripts to train transformers (_finetuneutils.py_ and _torchutils.py_) from Huggingface library with Pytorch.
We load the data from module _datautils.py_. We prepared the _main.py_ script to run the training and evaluation.
Finally, we used the _run_loop.py_ to run iteratively all the configuration for BET experiment:
datasets, transformers and different augmentation languages.

## Install dependencies

    pip install -r requirements.txt
