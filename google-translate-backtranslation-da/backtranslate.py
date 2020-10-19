import pandas as pd
from google.cloud import translate
import html
import numpy as np
import time
from dotenv import load_dotenv
load_dotenv()


def translate_text(target: str, text: list, noreverse=True):
    """Translates text into the target language.
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    time.sleep(0.5)  # Delay to avoid issues.

    translate_client = translate.Client()

    result = translate_client.translate(
        text,
        target_language=target if noreverse else 'en',
        source_language='en' if noreverse else target
    )

    return_list = list()
    for r in result:
        return_list.append(html.unescape(r['translatedText']))

    return return_list


def batch_sentences(sentences: list, n=50):
    """
    Batch a list into a list of list of n items.
    :param sentences: list of str.
    :param n: number of items in each batch list (int).
    :return: list of list of str.
    """
    sentence_batches = list()

    for i in range(int(np.ceil(len(sentences)/n))):
        sentence_batches.append(sentences[(n * i): (n*(i + 1))])

    return sentence_batches


def flatten_batches(batches: list):
    """
    Put on one level a list of list inside a unique list.
    :param batches: list of list of string.
    :return: list of string.
    """
    return [b for B in batches for b in B]


def batch_translate(batches: list, lang: str, noreverse=True):
    """
    Translate into batches.
    :param batches: list of list of str.
    :param lang: str with code of language.
    :param noreverse: boolean.
    :return: batch translated (list of list of str).
    """
    temp = [translate_text(lang, b, noreverse=noreverse) for b in batches]
    return temp


# Need to provide in environ variables your GOOGLE_APPLICATION_CREDENTIALS.

# CURRENT LANGUAGES :
# Chinese    zh
# Spanish    es
# Arabic     ar
# Japanese   ja
# Telugu     te
# Javanese   jv
# Korean     ko
# Vietnamese vi
# Turkish    tr
# Yoruba     yo

translation_labels = ['te', 'zh', 'es', 'ar', 'ja', 'jv', 'ko', 'vi', 'tr', 'yo']

key_paraphrase = "paraphrase"
filename = "downsampled100_tpc_train"

# Load data with paraphrase column.
data = pd.read_csv("%s.csv" % filename, index_col=0, encoding="utf8")

assert key_paraphrase in data.columns, "Change or verify your key for paraphrases match '%s'." % key_paraphrase

# Prepare temporary columns to concat directly to generated paraphrases.
data_temp = data
data_temp = data_temp.drop([key_paraphrase], axis=1)
data_temp = data_temp.reset_index()

# Add lang column with first english label: "en".
data["lang"] = "en"

data_aug = data

# VERIFY YOUR DATA BEFORE AUGMENTATION TO AVOID UNNECESSARY COST.

if __name__ == "__main__":

    # Prepare batches.
    batches = batch_sentences(data[key_paraphrase].values.tolist())

    for t in translation_labels:
        print("Current augmentation language: %s" % t)

        print("Translate source")
        translations_first = batch_translate(batches, lang=t)

        print("Translate translated")
        translations_second = batch_translate(translations_first, lang=t, noreverse=False)

        df = pd.DataFrame({key_paraphrase: flatten_batches(translations_second)})  # Build dataframe from batches.
        new_df = pd.concat([data_temp, df], axis=1)  # Append info to generated paraphrases.
        new_df["lang"] = t  # Add language column.

        new_df.to_csv("temp/%s_augmented_%s.csv" % (filename, t), encoding="utf8")  # Keep copies if failure at some point.

        data_aug = pd.concat([data_aug, new_df], axis=0)  # Append to all augmented data.

        time.sleep(5)  # Delay to avoid issues.

    print(data_aug)
    data_aug.to_csv("%s_augmented.csv" % filename, encoding="utf8")
