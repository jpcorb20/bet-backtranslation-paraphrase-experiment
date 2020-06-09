from unicodedata import normalize
import pandas as pd
from sklearn.model_selection import train_test_split

RDN_SPLIT_NUMBER = 123


def read_tsv_mrpc(input_file):

    tab = []
    with open(input_file, "r", encoding="utf8") as f:
        next(f)
        for line in f:
            tab.append(line.strip().split('\t'))

    dfObj = pd.DataFrame(tab)

    df = pd.DataFrame({'IDS': [i for i in range(len(tab))],
                       'sentence': dfObj[3],
                       'paraphrase': dfObj[4],
                       'quality': dfObj[0].astype('int64')})

    return df


def rename_mrpc(df):
    rename = df.rename(columns={'IDS': 'IDS', '#1 String': 'sentence', '#2 String': 'paraphrase', 'Quality': 'quality'})
    df = rename[['IDS', 'sentence', 'paraphrase', 'quality', 'lang', 'index']]
    return df


def remove_duplicates(df):
    duplicateRowsDF = df[df.duplicated(['sentence', 'paraphrase'])]
    duplicateRowsDF = duplicateRowsDF[duplicateRowsDF['lang'] != 'en']
    return df[~df['IDS'].isin(duplicateRowsDF['IDS'])]


def prepare_mrpc_data(baseline=True, langs=None, downsample=False):

    if not baseline:
        file = pd.read_excel('full_paraphrase_augmented.xlsx')

        df = rename_mrpc(file)

        df = df[df["lang"].isin(langs)]  # Keep augmented language

        df = remove_duplicates(df)
    else:
        df = read_tsv_mrpc("msr_paraphrase_train.txt")

    if downsample:
        df_en = balanced_downsampling(df[df["lang"] == "en"])
        df_langs = df[(df["lang"] != "en") & (df["index"].isin(df_en["index"]))]
        df = pd.concat([df_en, df_langs])

    train, val = train_test_split(df, test_size=0.2, random_state=RDN_SPLIT_NUMBER)

    # Load test in any case.
    test = read_tsv_mrpc("msr_paraphrase_test.txt")

    train_size, val_size, test_size = len(train), len(val), len(test)

    train.to_csv('data/train.csv')
    val.to_csv('data/dev.csv')
    test.to_csv('data/test.csv')

    return train_size, val_size, test_size


def parse_tpc_df(df):
    df = df.rename(columns={0: 'sentence', 1: 'paraphrase', 2: 'quality', 3: 'url'})
    df["IDS"] = df.index
    df["quality"] = df["quality"].apply(lambda x: int(x[1]))
    return df[['IDS', 'sentence', 'paraphrase', 'quality']]


def parse_tpc_scores(df):
    q = df["quality"]
    df.loc[:, "quality"] = df["quality"].apply(lambda x: 1 if x >= 4 else 0)
    return df[q != 3]


def prepare_tpc_data(baseline=True, langs=None, downsample=False):
    if baseline and langs is None:
        df = pd.read_csv("Twitter_URL_Corpus_train.txt", delimiter="\t", header=None)
        df = parse_tpc_df(df)
        df = parse_tpc_scores(df)

        if downsample:
            df = balanced_downsampling(df)

        train, val = train_test_split(df, test_size=0.2, random_state=RDN_SPLIT_NUMBER)
    elif langs is not None and isinstance(langs, list):
        df = pd.read_csv("tpc100_augmented.csv")
        df = df.drop("Unnamed: 0", axis=1)
        df["IDS"] = list(range(len(df)))

        df = df[df["lang"].isin(langs)]  # Keep augmented language

        train = remove_duplicates(df)

        val = pd.read_csv("downsampled_tpc_dev.csv")
        val = val.drop("Unnamed: 0", axis=1)
        val["IDS"] = list(range(len(val)))

    # Load test in any case.
    test = pd.read_csv("Twitter_URL_Corpus_test.txt", delimiter="\t", header=None)
    test = parse_tpc_df(test)
    test = parse_tpc_scores(test)

    train_size, val_size, test_size = len(train), len(val), len(test)

    train.to_csv('data/train.csv')
    val.to_csv('data/dev.csv')
    test.to_csv('data/test.csv')

    return train_size, val_size, test_size


def balanced_downsampling(df, n_samples=100):
    """
    Downsample dataframe with balance binary class 0/1 in quality column.
    :param df: dataframe (pandas.DataFrame).
    :param n_samples: number of samples (int, optional).
    :return: dataframe (pandas.DataFrame).
    """
    P = df[df["quality"] == 1].sample(n=int(n_samples/2), random_state=RDN_SPLIT_NUMBER)
    NP = df[df["quality"] == 0].sample(n=int(n_samples/2), random_state=RDN_SPLIT_NUMBER)
    PNP = pd.concat([P, NP])
    return PNP


def rename_quora(df):
    rename = df.rename(columns={"id": 'IDS', "question1": 'sentence',
                              "question2": 'paraphrase', "is_duplicate": 'quality'})
    return rename[['IDS', 'sentence', 'paraphrase', 'quality']]


def normalize_quora(x):
    """
    Normalize string from Quora dataset to finetune model. The quotes are a hack to make it work and let the tokenizer
    from huggingface remove some bad formatting.
    :param x: str.
    :return: normalized str.
    """
    return '"%s"' % normalize("NFKC", str(x).replace('"', ''))


def prepare_quora_data(baseline=True, langs=None, downsample=False):
    if baseline and langs is None:
        df = pd.read_csv("questions.csv", encoding="utf8")
        df = rename_quora(df)
        df["sentence"] = df["sentence"].apply(lambda x: normalize_quora(x))
        df["paraphrase"] = df["paraphrase"].apply(lambda x: normalize_quora(x))

        train, test = train_test_split(df, test_size=0.2, random_state=RDN_SPLIT_NUMBER)

        if downsample:
            train = balanced_downsampling(train)
            train, val = train_test_split(train, test_size=0.2, random_state=RDN_SPLIT_NUMBER)
        else:
            train, val = train_test_split(train, test_size=float(len(test) / len(train)), random_state=RDN_SPLIT_NUMBER)

    elif langs is not None and isinstance(langs, list):
        df = pd.read_csv("quora100_augmented.csv")
        df = df.drop("Unnamed: 0", axis=1)
        df["IDS"] = list(range(len(df)))

        df = df[df["lang"].isin(langs)]  # Keep augmented language

        train = remove_duplicates(df)

        val = pd.read_csv("downsampled_quora_dev.csv")
        val = val.drop("Unnamed: 0", axis=1)
        val["IDS"] = list(range(len(val)))

        test = pd.read_csv("downsampled_quora_test.csv")
        test = test.drop("Unnamed: 0", axis=1)
        test["IDS"] = list(range(len(test)))

    train_size, val_size, test_size = len(train), len(val), len(test)

    train.to_csv('data/train.csv')
    val.to_csv('data/dev.csv')
    test.to_csv('data/test.csv')

    return train_size, val_size, test_size


def prepare_data(dataset="mrpc", baseline=True, langs=None, downsample=False):
    if dataset == "mrpc":
        return prepare_mrpc_data(baseline=baseline, langs=langs, downsample=downsample)
    elif dataset == "tpc":
        return prepare_tpc_data(baseline=baseline, langs=langs, downsample=downsample)
    elif dataset == "quora":
        return prepare_quora_data(baseline=baseline, langs=langs, downsample=downsample)
    else:
        print("dataset don't exist")
        exit(1)
