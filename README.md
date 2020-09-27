# BET: A Backtranslation Approach for Easy Data Augmentation in Transformer-based Paraphrase Identification Context

We bet on it and it worked. This repository includes the scripts to train transformers (_finetuneutils.py_ and _torchutils.py_) from Huggingface library with Pytorch.
We load the data from module _datautils.py_. We prepared the _main.py_ script to run the training and evaluation.
Finally, we use the _run_loop.py_ to run iteratively all the configuration for BET experiment:
datasets, transformers and different augmentation languages.

# Clustering Languages 

We clustered all the Google Translation languages into the related language families based on the information provided in the Wikipedia info-boxes. The Romance branch is illustrated in the following figure.

![Romance Languages](img/language_family_tree.png) 


# Backtranslation
![backtranslation data augmentation scheme](img/aug_data.png) 


# Results

|  Model  |   Data   |  Acc  |   F1  | Precision | Recall |
|:-------:|:--------:|:-----:|:-----:|:---------:|:------:|
|   *BERT*  | baseline | 0.802 | 0.858 |   0.820   |  0.899 |
|   BERT  |    all   | 0.824 | 0.877 |   0.819   |  0.945 |
|   BERT  |    es    | 0.835 | 0.882 |   0.840   |  0.929 |
|  XLNet  | baseline | 0.845 | 0.886 |   0.868   |  0.905 |
|  XLNet  |    all   | 0.837 | 0.883 |   0.840   |  0.932 |
|  XLNet  |    ja    | 0.860 | 0.897 |   0.877   |  0.919 |
| RoBERTa | baseline | 0.874 | 0.906 |   0.898   |  0.914 |
| RoBERTa |    all   | 0.872 | 0.907 |   0.877   |  0.939 |
| RoBERTa |    vi    | 0.886 | 0.915 |   0.906   |  0.925 |
|  ALBERT | baseline | 0.853 | 0.890 |   0.885   |  0.895 |
|  ALBERT |    all   | 0.841 | 0.886 |   0.847   |  0.929 |
|  ALBERT |    yo    | 0.867 | 0.902 |   0.884   |  0.922 |



## Install dependencies

    pip install -r requirements.txt

## Disclaimer

Some parts of the code were inspired by HuggingFace Transformers Implementations.
