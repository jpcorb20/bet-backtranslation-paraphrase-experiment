# -*- coding: utf-8 -*-
import sys
from datautils import prepare_data
from finetuneutils import RDN_NUMBER_SEED, set_seed, process_model

LANGUAGES = ["en", "es", "ko", "zh", "ja", "jv", "te", "ar", "tr", "yo", "vi"]
LANGUAGES_LISTS = {L: ["en", L] for L in LANGUAGES}
LANGUAGES_LISTS.update({"all": LANGUAGES})

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("You don't have four arguments. Your command should look like: python main.py <model> <dataset> <lang>.")
        exit(1)

    _, model_type, dataset, lang = sys.argv
    set_seed(RDN_NUMBER_SEED)
    train_size, _, _ = prepare_data(dataset=dataset, langs=LANGUAGES_LISTS[lang], baseline=False)
    process_model(model_type, train_size, lang, dataset)
    exit(0)
