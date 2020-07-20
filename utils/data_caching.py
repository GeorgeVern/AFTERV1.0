import os

import torch
import logging
from torch.utils.data import TensorDataset

from utils.general_processors import processors, output_modes, standard_dev_tasks, convert_examples_to_features

logger = logging.getLogger(__name__)


def cache_examples(args, task, tokenizer, test=False):
    processor = processors[task]()
    output_mode = output_modes[task]

    modes = ["test"] if test else ["train", "dev"]
    cached_features_files = {}

    for mode in modes:
        cached_data_name = "cached_{}_{}_{}_{}".format(mode,
                                                       list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                       str(args.max_seq_length),
                                                       str(task))
        if mode != "test" and task not in standard_dev_tasks:
            cached_data_name += "_{}".format(args.seed)

        cached_features_files[mode] = os.path.join(args.data_dir, cached_data_name)

    if os.path.exists(cached_features_files[mode]) and not args.overwrite_cache:
        logger.info("Features are already cached in %s", cached_features_files[mode])
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        dataset_splits = {}
        if test:
            dataset_splits["test"] = processor.get_test_examples(args.data_dir)
        else:
            if task in standard_dev_tasks:
                dataset_splits["train"] = processor.get_train_examples(args.data_dir)
                dataset_splits["dev"] = processor.get_dev_examples(args.data_dir)
            else:
                dataset_splits["train"], dataset_splits["dev"] = processor.get_train_dev_examples(args.data_dir,
                                                                                                  args.seed)
        for mode, examples in dataset_splits.items():
            features = convert_examples_to_features(
                examples,
                tokenizer,
                label_list=label_list,
                max_length=args.max_seq_length,
                output_mode=output_mode,
                pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            )
            if args.local_rank in [-1, 0]:
                logger.info("Saving features into cached file %s", cached_features_files[mode])
                torch.save(features, cached_features_files[mode])

    return


def cache_after_datasets(args, main, aux, tokenizer, test=False):
    """
    This function is an extension of data caching for multi-task with multiple datasets and preprocessors
    :param args:
    :param main:
    :param aux:
    :param tokenizer:
    :param test:
    :return:
    """
    main_processor = processors[main]()
    aux_processor = processors[aux]()
    after_processors = [main_processor, aux_processor]
    tasks = [main, aux]
    output_mode = output_modes[main]

    for i, processor in enumerate(after_processors):
        modes = ["test"] if test else ["train", "dev"]
        cached_features_files = {}
        data_cache_dir = os.path.join("/".join(args.data_dir.split("/")[:-1]),  "AFTER/", main)

        for mode in modes:
            cached_data_name = "cached_{}_{}_{}_{}".format(mode,
                                                           list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                           str(args.max_seq_length),
                                                           str(tasks[i]))
            if mode != "test" and tasks[i] not in standard_dev_tasks:
                cached_data_name += "_{}".format(args.seed)

            cached_features_files[mode] = os.path.join(data_cache_dir, cached_data_name)

        if os.path.exists(cached_features_files[mode]) and not args.overwrite_cache:
            logger.info("Features are already cached in %s", cached_features_files[mode])
        else:
            os.makedirs(data_cache_dir, exist_ok=True)
            logger.info("Creating features from dataset file at %s", data_cache_dir)
            label_list = processor.get_labels()
            if tasks[i] in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
                # HACK(label indices are swapped in RoBERTa pretrained model)
                label_list[1], label_list[2] = label_list[2], label_list[1]
            dataset_splits = {}
            if test:
                dataset_splits["test"] = processor.get_test_examples(args.data_dirs[i])
            else:
                if tasks[i] in standard_dev_tasks:
                    dataset_splits["train"] = processor.get_train_examples(args.data_dirs[i], dom=i)
                    dataset_splits["dev"] = processor.get_dev_examples(args.data_dirs[i], dom=i)
                else:
                    dataset_splits["train"], dataset_splits["dev"] = processor.get_train_dev_examples(args.data_dirs[i],
                                                                                                      args.seed,
                                                                                                      dom=i)
            for mode, examples in dataset_splits.items():
                features = convert_examples_to_features(
                    examples,
                    tokenizer,
                    label_list=label_list,
                    max_length=args.max_seq_length,
                    output_mode=output_mode,
                    pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
                    pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                    pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
                    multilabel=True,
                )

                if args.local_rank in [-1, 0]:
                    logger.info("Saving features into cached file %s", cached_features_files[mode])
                    torch.save(features, cached_features_files[mode])

    return


def load_examples(args, task, mode):
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_data_name = "cached_{}_{}_{}_{}".format(mode,
                                                   list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                   str(args.max_seq_length),
                                                   str(task))
    if mode != "test" and task not in standard_dev_tasks:
        cached_data_name += "_{}".format(args.seed)
    cached_features_file = os.path.join(
        args.data_dir,
        cached_data_name
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        raise ValueError("Cached features have not been found in ".format(cached_features_file))

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # AfterBert
    all_num_tokens = []
    for input_id in list(all_input_ids):
        all_num_tokens.append(len(input_id.nonzero()))
    print("Max number of tokens in a sequence :---{}---".format(max(all_num_tokens)))
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def load_after_examples(args, main, aux, mode):
    """
    This function is an extension of load_examples for multi-task setting with multiple datasets and preprocessors
    :param args:
    :param main:
    :param aux:
    :return:
    """
    tasks = [main, aux]
    output_mode = output_modes[main]
    # the size of the smallest dataset (main dataset)
    len_dataset = 0
    datasets = []

    for i, task in enumerate(tasks):
        cached_features_files = {}
        data_cache_dir = "/".join(args.data_dir.split("/")[:-1])

        cached_data_name = "cached_{}_{}_{}_{}".format(mode,
                                                       list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                       str(args.max_seq_length),
                                                       str(task))
        if mode != "test" and task not in standard_dev_tasks:
            cached_data_name += "_{}".format(args.seed)
        cached_features_file = os.path.join(data_cache_dir, "AFTER/", main, cached_data_name)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            if i == 0:
                len_dataset = len(features)
            else:
                features = features[:len_dataset]
        else:
            raise ValueError("Cached features have not been found in ".format(cached_features_file))

        # Convert to Tensors and build dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        # AfterBert
        all_num_tokens = []
        for input_id in list(all_input_ids):
            all_num_tokens.append(len(input_id.nonzero()))
        print("Max number of tokens in a sequence :---{}---".format(max(all_num_tokens)))
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        if output_mode == "classification":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        elif output_mode == "regression":
            all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

        datasets.append(TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels))
    return datasets[0], datasets[1]


def load_domain_examples(args, main, aux, mode):
    """
    This function is an extension of load_examples for multi-task setting with multiple datasets and preprocessors
    :param args:
    :param main:
    :param aux:
    :return:
    """
    tasks = [main, aux]
    # the size of the smallest dataset (main dataset)
    len_dataset = 0
    all_input_ids, all_attention_mask, all_token_type_ids, all_labels = [], [], [], []

    for i, task in enumerate(tasks):
        data_cache_dir = "/".join(args.data_dir.split("/")[:-1])

        cached_data_name = "cached_{}_{}_{}_{}".format(mode,
                                                       list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                                       str(args.max_seq_length),
                                                       str(task))
        if mode != "test" and task not in standard_dev_tasks:
            cached_data_name += "_{}".format(args.seed)
        cached_features_file = os.path.join(data_cache_dir, "AFTER/", main, cached_data_name)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
            if i == 0:
                len_dataset = len(features)
            else:
                features = features[:len_dataset]
        else:
            raise ValueError("Cached features have not been found in ".format(cached_features_file))

        # Convert to Tensors and build dataset
        all_input_ids.append(torch.tensor([f.input_ids for f in features], dtype=torch.long))
        # AfterBert
        all_num_tokens = []
        for input_id in list(all_input_ids[i]):
            all_num_tokens.append(len(input_id.nonzero()))
        print("Max number of tokens in a sequence :---{}---".format(max(all_num_tokens)))
        all_attention_mask.append(torch.tensor([f.attention_mask for f in features], dtype=torch.long))
        all_token_type_ids.append(torch.tensor([f.token_type_ids for f in features], dtype=torch.long))
        all_labels.append(torch.tensor([f.label[1] for f in features], dtype=torch.long))

    dataset = TensorDataset(torch.cat(all_input_ids), torch.cat(all_attention_mask), torch.cat(all_token_type_ids),
                            torch.cat(all_labels))
    return dataset
