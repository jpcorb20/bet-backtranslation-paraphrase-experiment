import os
import sys
import csv
import pandas as pd

csv.field_size_limit(2147483647)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(str(cell, encoding='utf-8') for cell in line)
                lines.append(line)
            return lines


class BinaryProcessor(DataProcessor):
    def __init__(self):
        self.train_file = "train.csv"
        self.dev_file = "dev.csv"
        self.test_file = "test.csv"

    def get_train_examples(self, data_dir):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, self.train_file))
        return self._create_examples(df)

    def get_dev_examples(self, data_dir):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, self.dev_file))
        return self._create_examples(df)

    def get_test_examples(self, data_dir):
        """See base class."""
        df = pd.read_csv(os.path.join(data_dir, self.test_file))
        return self._create_examples(df)

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, input):
        examples = []
        for id, ta, tb, l in zip(list(input["IDS"]), list(input['sentence']), list(input['paraphrase']),
                                 list(input['quality'])):
            examples.append(InputExample(guid=id,
                                         text_a=ta,
                                         text_b=tb,
                                         label=l))
        return examples