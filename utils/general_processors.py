""" Data processors and helpers """

import logging
import os
import csv

import json
from transformers.data.processors.utils import DataProcessor, InputExample, InputFeatures
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def convert_examples_to_features(
        examples,
        tokenizer,
        max_length=512,
        task=None,
        label_list=None,
        output_mode=None,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        mask_padding_with_zero=True,
        multilabel=False,
):
    """
    Loads a data file into a list of ``InputFeatures``

    Args:
        examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
        tokenizer: Instance of a tokenizer that will tokenize the examples
        max_length: Maximum example length
        task: GLUE task
        label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
        output_mode: String indicating the output mode. Either ``regression`` or ``classification``
        pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
        pad_token: Padding token
        pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
        mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
            and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
            actual values)

    Returns:
        If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
        containing the task-specific features. If the input is a list of ``InputExamples``, will return
        a list of task-specific ``InputFeatures`` which can be fed to the model.

    """

    if task is not None:
        processor = glue_processors[task]()
        if label_list is None:
            label_list = processor.get_labels()
            logger.info("Using label list %s for task %s" % (label_list, task))
        if output_mode is None:
            output_mode = glue_output_modes[task]
            logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}
    if multilabel:
        domain_label_map = {label: i for i, label in enumerate(["0", "1"])}

    features = []
    for (ex_index, example) in enumerate(examples):
        len_examples = len(examples)
        if ex_index % 10000 == 0:
            logger.info("Writing example %d/%d" % (ex_index, len_examples))

        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label] if not multilabel else label_map[example.label[0]]
        elif output_mode == "regression":
            label = float(example.label) if not multilabel else float(example.label[0])
        else:
            raise KeyError(output_mode)
        if multilabel:
            label = [label, domain_label_map[example.label[1]]]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            if multilabel:
                logger.info("label: %s (id = %d)" % (example.label[0], label[0]))
                logger.info("domain label: %s (id = %d)" % (example.label[1], label[1]))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )

    return features


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["premise"].numpy().decode("utf-8"),
            tensor_dict["hypothesis"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")), "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliMismatchedProcessor(MnliProcessor):
    """Processor for the MultiNLI Mismatched data set (GLUE version)."""

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev_mismatched.tsv")), "dev_matched")


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class Sst2Processor(DataProcessor):
    """Processor for the SST-2 data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class StsbProcessor(DataProcessor):
    """Processor for the STS-B data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_labels(self):
        """See base class."""
        return [None]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[7]
            text_b = line[8]
            label = line[-1]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QqpProcessor(DataProcessor):
    """Processor for the QQP data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question1"].numpy().decode("utf-8"),
            tensor_dict["question2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            try:
                text_a = line[3]
                text_b = line[4]
                label = line[5]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class QnliProcessor(DataProcessor):
    """Processor for the QNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["question"].numpy().decode("utf-8"),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev_matched", dom)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class RteProcessor(DataProcessor):
    """Processor for the RTE data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_labels(self):
        """See base class."""
        return ["entailment", "not_entailment"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class WnliProcessor(DataProcessor):
    """Processor for the WNLI data set (GLUE version)."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[1]
            text_b = line[2]
            label = line[-1]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


standard_dev_tasks = ["cola", "mnli", "mrpc", "sst-2", "sts-b", "qqp", "qnli", "rte", "wnli", "amz-el", "pubmed",
                      "math", "compsci"]


class Trec6Processor(DataProcessor):
    """Processor for the TREC-6 data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def read_txt(cls, input_file):
        """Reads a text file."""
        return open(input_file, "r", encoding="utf-8").readlines()

    def get_train_dev_examples(self, data_dir, seed, split=0.1, dom=-1):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self.read_txt(os.path.join(data_dir, "train.txt"))
        for line in lines:
            columns = line.rstrip().split(" ")
            Y.append(columns[0].split(":")[0])
            X.append(" ".join(columns[1:]))
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
        train_examples = self._create_examples(zip(X_train, Y_train), "train", dom)
        dev_examples = self._create_examples(zip(X_dev, Y_dev), "dev", dom)
        return train_examples, dev_examples

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AgnewsProcessor(DataProcessor):
    """Processor for the AG NEWS data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter=",", quotechar=quotechar))

    def get_train_dev_examples(self, data_dir, seed, split=0.1, dom=-1):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_csv(os.path.join(data_dir, "train.csv"))
        for line in lines:
            Y.append(line[0].replace('"', ""))
            X.append(",".join(line[1:]).rstrip())
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
        train_examples = self._create_examples(zip(X_train, Y_train), "train", dom)
        dev_examples = self._create_examples(zip(X_dev, Y_dev), "dev", dom)
        return train_examples, dev_examples

    def get_labels(self):
        """See base class."""
        return ["1", "2", "3", "4"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class ImdbProcessor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_dev_examples(self, data_dir, seed, split=0.1, dom=-1):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        for line in lines[1:]:
            columns = "".join(line).split(",")
            Y.append(columns[-1])
            X.append(",".join(columns[:-1]).rstrip().replace('"', ""))
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
        train_examples = self._create_examples(zip(X_train, Y_train), "train", dom)
        dev_examples = self._create_examples(zip(X_dev, Y_dev), "dev", dom)
        return train_examples, dev_examples

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class EuroparlProcessor(DataProcessor):
    """Processor for the IMDB data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def _read_csv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\n", quotechar=quotechar))

    def get_train_dev_examples(self, data_dir, seed, split=0.1, dom=-1):
        """Splits train set into train and dev sets."""
        X, Y = [], []
        lines = self._read_csv(os.path.join(data_dir, "train.csv"))
        for line in lines:
            Y.append("0")
            X.append(",".join(line).rstrip())
        X_train, X_dev, Y_train, Y_dev = train_test_split(X, Y, test_size=split, stratify=Y, random_state=seed)
        train_examples = self._create_examples(zip(X_train, Y_train), "train", dom)
        dev_examples = self._create_examples(zip(X_dev, Y_dev), "dev", dom)
        return train_examples, dev_examples

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "test.csv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples for the dev set."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class AmazonElectronicsProcessor(DataProcessor):
    """Processor for the AMZ-El data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["1.0", "2.0", "3.0", "4.0", "5.0"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a, label = line
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class PubMedProcessor(DataProcessor):
    """Processor for the PubMed data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def read_txt(cls, input_file):
        """Reads a text file."""
        return open(input_file, "r", encoding="utf-8").readlines()

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self.read_txt(os.path.join(data_dir, "train.txt")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self.read_txt(os.path.join(data_dir, "dev.txt")), "dev", dom)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self.read_txt(os.path.join(data_dir, "test.txt")), "test")

    def get_labels(self):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            d = json.loads(line)
            summary = "\n".join(d["abstract_text"])
            summary = summary.replace("<S>", "").replace("</S>", "")
            text_a, label = summary, "0"
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MathProcessor(DataProcessor):
    """Processor for the AMZ-El data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = "0"
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class CompSciProcessor(DataProcessor):
    """Processor for the AMZ-El data set."""

    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train", dom)

    def get_dev_examples(self, data_dir, dom=-1):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev", dom)

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0"]

    def _create_examples(self, lines, set_type, dom=-1):
        """Creates examples."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = "0"
            if dom != -1:
                label = [label, str(dom)]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


processors = {
    "cola": ColaProcessor,
    "mnli": MnliProcessor,
    "mnli-mm": MnliMismatchedProcessor,
    "mrpc": MrpcProcessor,
    "sst-2": Sst2Processor,
    "sts-b": StsbProcessor,
    "qqp": QqpProcessor,
    "qnli": QnliProcessor,
    "rte": RteProcessor,
    "wnli": WnliProcessor,
    "trec-6": Trec6Processor,
    "ag_news": AgnewsProcessor,
    "imdb": ImdbProcessor,
    "europarl": EuroparlProcessor,
    "amz-el": AmazonElectronicsProcessor,
    "pubmed": PubMedProcessor,
    "math": MathProcessor,
    "compsci": CompSciProcessor,
}

output_modes = {
    "cola": "classification",
    "mnli": "classification",
    "mnli-mm": "classification",
    "mrpc": "classification",
    "sst-2": "classification",
    "sts-b": "regression",
    "qqp": "classification",
    "qnli": "classification",
    "rte": "classification",
    "wnli": "classification",
    "trec-6": "classification",
    "ag_news": "classification",
    "imdb": "classification",
    "europarl": "classification",
    "amz-el": "classification",
    "pubmed": "classification",
    "math": "classification",
    "compsci": "classification",
}
