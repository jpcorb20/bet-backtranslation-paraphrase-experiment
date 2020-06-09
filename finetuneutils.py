import os
import gc
import json
import random
import pandas as pd
import numpy as np
from sklearn.metrics import *
from tqdm import tqdm, trange
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer,\
                         XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,\
                         RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,\
                         AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer,\
                         glue_convert_examples_to_features, AdamW, get_linear_schedule_with_warmup
from apex import amp
from torchutils import BinaryProcessor

RDN_NUMBER_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_CLASSES = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer)
}

MODEL_PARAMS = {
    'bert': {"lr": 3e-5, "bs": 32, "filename": "bert-base-uncased"},
    'xlnet': {"lr": 3e-5, "bs": 16, "filename": "xlnet-base-cased"},
    'roberta': {"lr": 3e-5, "bs": 32, "filename": "roberta-base"},
    'albert': {"lr": 3e-5, "bs": 32, "filename": "albert-base-v1"}
}


def set_seed(seed, gpus=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(RDN_NUMBER_SEED)
    if gpus:
        torch.cuda.manual_seed_all(seed)


def set_args(model_name, train_size, n_train_epochs):
    steps = int(np.floor(train_size / MODEL_PARAMS[model_name]["bs"]))
    return {
        'data_dir': 'data/',
        'n_gpu': 1,
        'max_steps': -1,
        'model_type': model_name,
        'model_name': MODEL_PARAMS[model_name]["filename"],
        'task_name': 'binary',
        'output_dir': 'outputs/',
        'cache_dir': 'cache/',
        'do_train': True,
        'do_eval': True,
        'do_predict': True,
        'fp16': True,
        'fp16_opt_level': 'O1',
        'max_seq_length': 128,
        'output_mode': 'classification',
        'train_batch_size': MODEL_PARAMS[model_name]["bs"],
        'eval_batch_size': MODEL_PARAMS[model_name]["bs"],
        'local_rank': -1,
        'gradient_accumulation_steps': 1,
        'num_train_epochs': n_train_epochs,
        'weight_decay': 0,
        'learning_rate': MODEL_PARAMS[model_name]["lr"],
        'adam_epsilon': 1e-8,
        'warmup_steps': round(0.1 * steps),
        'max_grad_norm': 1.0,
        'logging_steps': steps,
        'evaluate_during_training': True,
        'save_steps': steps,
        'eval_all_checkpoints': True,
        'per_gpu_eval_batch_size': MODEL_PARAMS[model_name]["bs"],  # same as batchsize
        'per_gpu_train_batch_size': MODEL_PARAMS[model_name]["bs"],
        'overwrite_output_dir': True,
        'reprocess_input_data': True,
        'notes': 'paraphrase task'
    }


def create_training_config(args):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args["model_type"]]
    config = config_class.from_pretrained(args['model_name'], num_labels=2, finetuning_task='binary')
    tokenizer = tokenizer_class.from_pretrained(args['model_name'], do_lower_case=True)
    model = model_class.from_pretrained(args['model_name'])
    model.to(DEVICE)

    return model, tokenizer, config


def process_train(args, model, tokenizer):
    if args['do_train']:
        train_dataset = load_and_cache_examples(tokenizer, args, training=True, evaluate=False, predict=False)
        train(train_dataset, model, tokenizer, args)


def create_prediction_config(args):
    folder_name = './outputs/checkpoint-%d/' % (args['num_train_epochs'] * args['save_steps'],)

    local_config, local_classification, local_tokenizer = MODEL_CLASSES[args["model_type"]]
    config = local_config.from_pretrained(folder_name, num_labels=2, finetuning_task='binary')
    model = local_classification.from_pretrained(folder_name)  # re-load
    tokenizer = local_tokenizer.from_pretrained(folder_name, do_lower_case=True)  # re-load
    model.to(DEVICE)

    return model, tokenizer, config


def load_and_cache_examples(tokenizer, args, training=False, evaluate=False, predict=False):
    processor = BinaryProcessor()
    output_mode = args['output_mode']

    # mode = 'dev' if evaluate else 'train'
    if predict:
        mode = 'test'
    if evaluate:
        mode = 'dev'
    if training:
        mode = 'train'
    cached_features_file = os.path.join(args['data_dir'],
                                        f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_binary")

    if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args['data_dir'])
        if training:
            examples = processor.get_train_examples(args['data_dir'])
        if predict:
            examples = processor.get_test_examples(args['data_dir'])

        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args['max_seq_length'],
            output_mode=output_mode)

        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([(f.token_type_ids if f.token_type_ids is not None else 0) for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def train(train_dataset, model, tokenizer, args):
    """ Train the model """

    args['train_batch_size'] = args['per_gpu_train_batch_size'] * max(1, args['n_gpu'])
    train_sampler = RandomSampler(train_dataset) if args['local_rank'] == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])

    if args['max_steps'] > 0:
        t_total = args['max_steps']
        args['num_train_epochs'] = args['max_steps'] // (
                len(train_dataloader) // args['gradient_accumulation_steps']) + 1
    else:
        t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args['weight_decay'],
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args['model_name'], "optimizer.pt")) and os.path.isfile(
            os.path.join(args['model_name'], "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args['model_name'], "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args['model_name'], "scheduler.pt")))

    if args['fp16']:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])

    # multi-gpu training (should be after apex fp16 initialization)
    if args['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args['local_rank'] != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args['local_rank']], output_device=args['local_rank'], find_unused_parameters=True,
        )

    # Train!

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args['model_name']):
        # set global_step to global_step of last saved checkpoint from model path
        try:
            global_step = int(args['model_name'].split("-")[-1].split("/")[0])
        except ValueError:
            global_step = 0
        epochs_trained = global_step // (len(train_dataloader) // args['gradient_accumulation_steps'])
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args['gradient_accumulation_steps'])

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args['num_train_epochs']), desc="Epoch", disable=args['local_rank'] not in [-1, 0],
    )

    set_seed(RDN_NUMBER_SEED)  # Added here for reproductibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args['local_rank'] not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(DEVICE) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args['model_type'] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args['model_type'] in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args['n_gpu'] > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                if args['fp16']:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['local_rank'] in [-1, 0] and args['logging_steps'] > 0 and global_step % args[
                    'logging_steps'] == 0:
                    logs = {}
                    if (
                            args['local_rank'] == -1 and args['evaluate_during_training']
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(model, tokenizer, args)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args['logging_steps']
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    print(json.dumps({**logs, **{"step": global_step}}))

                if args['local_rank'] in [-1, 0] and args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

            if args['max_steps'] > 0 and global_step > args['max_steps']:
                epoch_iterator.close()
                break
        if args['max_steps'] > 0 and global_step > args['max_steps']:
            train_iterator.close()
            break

    return global_step, tr_loss / global_step


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {"acc": simple_accuracy(preds, labels)}


def evaluate(model, tokenizer, args, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args['task_name'] == "mnli" else (args['task_name'],)
    eval_outputs_dirs = (args['output_dir'], args['output_dir'] + "-MM") if args['task_name'] == "mnli" else (
        args['output_dir'],)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(tokenizer, args, training=False, evaluate=True, predict=False)

        if not os.path.exists(eval_output_dir) and args['local_rank'] in [-1, 0]:
            os.makedirs(eval_output_dir)

        args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

        # multi-gpu eval
        if args['n_gpu'] > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(DEVICE) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args['model_type'] != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args['model_type'] in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args['output_mode'] == "classification":
            preds = np.argmax(preds, axis=1)
        elif args['output_mode'] == "regression":
            preds = np.squeeze(preds)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            for key in sorted(result.keys()):
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results


def all_metrics_computation(predicted_y, display=True):
    """
    Compute every metric possible on test set.
    :param predicted_y: model predictions.
    :param display: do we want to print.
    :return: result dictionary (dict).
    """
    # Load gold standard.
    test_set = pd.read_csv("data/test.csv", index_col=0)
    test_y = test_set['quality'].tolist()

    results = {
        "accuracy": accuracy_score(test_y,  predicted_y),
        "macro_f1_score": f1_score(test_y,  predicted_y, average='macro'),
        "weighted_f1_score": f1_score(test_y,  predicted_y, average='weighted'),
        "f1_score": f1_score(test_y,  predicted_y),
        "precision_score": precision_score(test_y,  predicted_y),
        "recall_score": recall_score(test_y,  predicted_y),
        "ROC-AUC": roc_auc_score(test_y,  predicted_y)
    }

    if display:
        print("******************Evaluation***********************")
        for k, v in results.items():
            print("%20s: %0.6f" % (k, v))

    return results


def save_perf(preds, lang, dataset, args):
    perf = all_metrics_computation(preds)

    if os.path.exists("%s_100downsamples_results.json" % dataset):
        with open("%s_100downsamples_results.json" % dataset, "r") as fp:
            results = json.load(fp)
    else:
        results = list()

    results.append({"config": {"model": args["model_type"], "langs": lang, "dataset": dataset}, "results": dict(**perf)})

    with open("%s_100downsamples_results.json" % dataset, "w") as fp:
        json.dump(results, fp)


def process_prediction(model, tokenizer, args, lang, dataset):
    eval_dataset = load_and_cache_examples(tokenizer, args, training=False, evaluate=False, predict=True)

    args['eval_batch_size'] = args['per_gpu_eval_batch_size'] * max(1, args['n_gpu'])
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args['eval_batch_size'])

    # multi-gpu eval
    if args['n_gpu'] > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    preds = None
    for batch in tqdm(eval_dataloader, desc="Predicting"):
        model.eval()

        batch = tuple(t.to(DEVICE) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args['model_type'] != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args['model_type'] in ["bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(**inputs)

        if preds is None:
            preds = outputs[:2][1].to('cpu').numpy()
        else:
            preds = np.append(preds, outputs[:2][1].to('cpu').numpy(), axis=0)

        del outputs
        torch.cuda.empty_cache()

    if preds is not None:
        preds = np.argmax(preds, axis=1)
        save_perf(preds, lang, dataset, args)


def print_gpu():
    print('Allocated:', round(torch.cuda.memory_allocated(DEVICE)/1024**3, 1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(DEVICE)/1024**3, 1), 'GB')


def debug_gpu():
    # Debug out of memory bugs.
    tensor_list = []
    for obj in gc.get_objects():
        if torch.is_tensor(obj):
            tensor_list.append(obj)
    print(f'Count of tensors: {len(tensor_list)}.')


def reset_gpu():
    print_gpu()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()
    print_gpu()
    debug_gpu()


def process_model(model_type, train_size, lang, dataset, epochs=3):
    args = set_args(model_type, train_size, epochs)

    # TRAINING
    model, tokenizer, config = create_training_config(args)
    process_train(args, model, tokenizer)
    del model, tokenizer, config
    reset_gpu()

    # EVAL
    model_p, tokenizer_p, config_p = create_prediction_config(args)
    process_prediction(model_p, tokenizer_p, args, lang, dataset)
    del model_p, tokenizer_p, config_p, args
    reset_gpu()
