import os
from finetuneutils import MODEL_PARAMS

DATASETS = ["mrpc", "tpc", "quora"]
LANGUAGES = ["es", "ko", "zh", "ja", "jv", "te", "ar", "tr", "yo", "vi", "all"]  # "en",

if __name__ == "__main__":
    for dataset in DATASETS:
        print(dataset)
        for model_type in MODEL_PARAMS.keys():
            print(model_type)
            for lang in LANGUAGES:
                print(lang)
                os.system("python main.py %s %s %s" % (model_type, dataset, lang))
