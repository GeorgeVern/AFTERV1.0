
import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from utils.utils_exams import convert_examples_to_features, processors
logger = logging.getLogger(__name__)


def select_field(features, field):
    return [[choice[field] for choice in feature.choices_features] for feature in features]


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = (
        processors[args.new_task_name]()
        if args.new_task_name not in ["arc", "exams"]
        else processors[args.new_task_name](para_type=args.para_type)
    )
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = "dev"
    elif test:
        cached_mode = "test"
    else:
        cached_mode = "train"
    assert not (evaluate and test)
    cached_features_file = os.path.join(
        args.new_data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            "__".join(list(filter(None, args.model_name_or_path.split("/")))[-2:]),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.new_data_dir)
        elif test:
            examples = processor.get_test_examples(args.new_data_dir)
        else:
            examples = processor.get_train_examples(args.new_data_dir)
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token_segment_id=tokenizer.pad_token_type_id,
            kb_masking=bool(args.model_type in ["bert-kb", "roberta-kb", "xlm-roberta-kb"]),
        )
    if args.local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, "input_mask"), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, "segment_ids"), dtype=torch.long)
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    all_answers_masks = torch.tensor([f.answers_mask for f in features], dtype=torch.long)
    all_domain_labels = torch.tensor([0 for i in range(len(features))], dtype=torch.int)
    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_answers_masks,
            all_domain_labels)

    if evaluate or test:
        all_example_ids = [x.example_id for x in features]

        return dataset, all_example_ids

    return dataset
