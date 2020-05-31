""" Finetuning the AFTER models for sequence classification (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

import argparse
import json
import logging
import os
import random

import glob

import csv
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    AdamW,
    BertConfig,
    BertTokenizer,
    get_linear_schedule_with_warmup, XLNetConfig, XLNetTokenizer, WEIGHTS_NAME,
)

from after_models.after_bert import AfterBertForSequenceClassification
from after_models.after_xlnet import AfterXLNetForSequenceClassification
from sys_config import DATA_DIR
from utils.data_caching import cache_after_datasets, load_after_examples
from utils.metrics import compute_metrics
from utils.general_processors import processors, output_modes

from utils.config import train_options

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
        BertConfig,
    )
    ),
    (),
)

MODEL_CLASSES = {
    "afterbert": (BertConfig, AfterBertForSequenceClassification, BertTokenizer),
    "afterxlnet": (XLNetConfig, AfterXLNetForSequenceClassification, XLNetTokenizer)
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_main_dataset, train_aux_dataset, model, tokenizer):
    """ Train the model """

    # The batch size must be halved in order to fit one batch from each domain
    args.train_batch_size = (args.per_gpu_train_batch_size * max(1, args.n_gpu)) // 2
    train_main_sampler = RandomSampler(train_main_dataset) if args.local_rank == -1 else DistributedSampler(
        train_main_dataset)
    train_main_dataloader = DataLoader(train_main_dataset, sampler=train_main_sampler, batch_size=args.train_batch_size)

    train_aux_sampler = RandomSampler(train_aux_dataset) if args.local_rank == -1 else DistributedSampler(
        train_aux_dataset)
    train_aux_dataloader = DataLoader(train_aux_dataset, sampler=train_aux_sampler, batch_size=args.train_batch_size)

    # # the length of the dataloader has been effectively doubled
    # len_dataloader = len(train_main_dataloader) + len(train_aux_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_main_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_main_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # AfterBERT
    args.logging_steps = len(train_main_dataloader) // args.num_evals
    args.warmup_steps = args.warmup_proportion * t_total

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                      correct_bias=args.bias_correction)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num of Main examples = %d", len(train_main_dataset))
    logger.info("  Num of Auxiliary examples = %d", len(train_aux_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_main_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_main_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    # tr_loss, logging_loss = 0.0, 0.0
    main_tr_loss, dom_tr_loss, main_logging_loss, dom_logging_loss = 0.0, 0.0, 0.0, 0.0
    # AfterBERT
    best_val_loss = 1e5

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(zip(train_main_dataloader, train_aux_dataloader), total=len(train_main_dataloader),
                              desc="Iteration",
                              disable=args.local_rank not in [-1, 0])
        for step, multi_batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()

            if args.lambd_anneal:
                p = float(global_step / 100 / t_total)
                alpha = 2. / (1. + np.exp(-15 * p)) - 1
                model.grl.lambd = alpha
                # print('GRL parameter is : {}'.format(alpha))

            for ind, batch in enumerate(multi_batch):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                is_aux = True if ind == 1 else False
                outputs = model(**inputs, aux=is_aux)

                if not is_aux:
                    # We either have both domain and main loss (Main dataset)
                    main_loss, dom_loss = outputs[:2]  # model outputs are always tuple in transformers (see doc)
                else:
                    # Or we only have domain loss (Auxiliary dataset)
                    dom_loss = torch.mean(torch.stack([dom_loss, outputs[1]]))

            loss = main_loss + dom_loss

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # tr_loss += loss.item()
            main_tr_loss += main_loss.item()
            dom_tr_loss += dom_loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                            args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    main_loss_scalar = (main_tr_loss - main_logging_loss) / args.logging_steps
                    dom_loss_scalar = (dom_tr_loss - dom_logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["main loss"] = main_loss_scalar
                    logs["dom loss"] = dom_loss_scalar
                    main_logging_loss, dom_logging_loss = main_tr_loss, dom_tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

                    # AfterBERT
                    if args.local_rank in [-1, 0] and results["loss"] < best_val_loss:
                        # AfterBERT
                        train_steps = global_step / len(train_main_dataloader)
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir,
                                                  "checkpoint".format(train_steps))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))
                        logger.info("Saving model checkpoint to %s", output_dir)

                        torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        logger.info("Saving optimizer and scheduler states to %s", output_dir)

                        # AfterBERT
                        best_val_loss = results["loss"]

                        output_ckpt_file = os.path.join(output_dir, "best_loss.txt")
                        with open(output_ckpt_file, "w+") as writer:
                            for key in sorted(results.keys()):
                                writer.write("%s = %s\n" % (key, str(results[key])))
                            writer.write("steps = %s\n" % (str(train_steps)))

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    return global_step, main_tr_loss / global_step


def evaluate(args, model, prefix="", save_preds=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "-MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset, _ = load_after_examples(args, eval_task, args.aux_name, mode="dev")

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss, dom_loss = 0.0, 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs, aux=False)
                tmp_eval_loss, tmp_dom_loss, logits = outputs[:3]

                eval_loss += tmp_eval_loss.mean().item()
                dom_loss += tmp_dom_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()[:, 0]
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy()[:, 0], axis=0)

        eval_loss = eval_loss / nb_eval_steps

        if args.eval_domain:
            _, eval_aux_dataset = load_after_examples(args, eval_task, args.aux_name, mode="dev")

            if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
                os.makedirs(eval_output_dir)

            args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
            # Note that DistributedSampler samples randomly
            eval_aux_sampler = SequentialSampler(eval_aux_dataset)
            eval_aux_dataloader = DataLoader(eval_aux_dataset, sampler=eval_aux_sampler,
                                             batch_size=args.eval_batch_size)

            # multi-gpu eval
            if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
                model = torch.nn.DataParallel(model)

            # Eval!
            logger.info("***** Running auxiliary evaluation {} *****".format(prefix))
            logger.info("  Num examples = %d", len(eval_aux_dataset))
            logger.info("  Batch size = %d", args.eval_batch_size)

            for batch in tqdm(eval_aux_dataloader, desc="Evaluating"):
                model.eval()
                batch = tuple(t.to(args.device) for t in batch)

                with torch.no_grad():
                    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                    if args.model_type != "distilbert":
                        inputs["token_type_ids"] = (
                            batch[2] if args.model_type in ["bert", "xlnet", "albert"] else None
                        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                    outputs = model(**inputs, aux=True)
                    _, tmp_dom_loss, logits = outputs[:3]

                    dom_loss += tmp_dom_loss.mean().item()
                nb_eval_steps += 1

            dom_loss = dom_loss / nb_eval_steps

        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)

        if save_preds:
            metrics = np.array(preds == out_label_ids, dtype=int)
            with open(eval_output_dir + '/preds.tsv', 'a+') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerow(["preds", "labels"])
                for pred, label, metric in zip(preds, out_label_ids, metrics):
                    tsv_writer.writerow([pred, label])

        result = compute_metrics(eval_task, preds, out_label_ids)
        # AfterBERT
        result.update({"loss": eval_loss})
        if args.eval_domain:
            result.update({"dom loss": dom_loss})
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "a+") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def main(args):
    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and args.do_train
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Prepare task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    # Prepare auxiliary dataset
    args.aux_name = args.auxiliary_name.lower()
    if args.aux_name not in processors:
        raise ValueError("Task not found: %s" % (args.aux_name))
    main_processor = processors[args.task_name]()

    args.output_mode = output_modes[args.task_name]
    label_list = main_processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    # find the checkpoint file in case of not training
    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
    )

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        # finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path if args.do_train else checkpoints[0],
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
        lambd=args.lambd,
        mean_pool=args.mean_pool
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # AfterBert
    cache_after_datasets(args, args.task_name, args.aux_name, tokenizer, test=False)
    # cache_after_datasets(args, args.task_name, args.aux_name, tokenizer, test=True)

    # Training
    if args.do_train:
        train_main_dataset, train_aux_dataset = load_after_examples(args, args.task_name, args.aux_name, mode="train")
        global_step, tr_loss = train(args, train_main_dataset, train_aux_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Evaluation
    logger.info("Evaluate the following checkpoint: %s", checkpoints[0])
    logs = {}

    prefix = checkpoints[0].split("/")[-1] if checkpoints[0].find("checkpoint") != -1 else ""

    results = evaluate(args, model, prefix=prefix, save_preds=True)

    for key, value in results.items():
        eval_key = "eval_{}".format(key)
        logs[eval_key] = value
    print(json.dumps({**logs}))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", required=False,
                        default='afterBert_finetune_sts-b_pubmed.yaml',
                        help="config file of input data")

    parser.add_argument("--seed", type=int, default=12, help="random seed for initialization")

    parser.add_argument("--lambd", type=float, default=-0.1, help="lambda hyperparameter for adversarial loss")

    parser.add_argument("--lambd_anneal", type=bool, default=False, help="lambda hyperparameter for adversarial loss")

    parser.add_argument("--mean_pool", type=bool, default=False,
                        help="Whether to use mean pooling of the output hidden states insted of CLS token for the domain classifier")

    parser.add_argument("--do_train", type=bool, default=True, help="Whether to run training.")

    parser.add_argument("--eval_domain", type=bool, default=False,
                        help="Evaluate domain loss on validation set (doubles validation time)")

    parser.add_argument("--no_cuda", type=bool, default=False, help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",
    )

    args = parser.parse_args()

    config = train_options(args.input)
    # Merge the input arguments with the configuration yaml
    args.__dict__.update(config)
    # Create one folder per seed
    if args.lambd_anneal:
        args.lambd = 0.00
    args.output_dir = "".join(
        (args.output_dir, "/AFTER_{}/seed_{}_lambda_{}".format(args.auxiliary_name, args.seed, args.lambd)))
    if args.mean_pool:
        args.output_dir += "_mean"
    args.data_dirs = [args.data_dir, "".join((DATA_DIR, config["auxiliary_name"]))]
    main(args)
