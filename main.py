import time
from typing import List, Dict

import tokenizers
from Levenshtein import distance as lev_distance
from tqdm import tqdm
from transformers import (
    pipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    IntervalStrategy,
    Seq2SeqTrainer,
)
import data_operations
from crypto_news_dataset import CryptoNewsDataset
import torch
import os

import gc

from metrics import build_compute_metrics_fn, calculate_rouge, calculate_bleu

gc.collect()

torch.cuda.empty_cache()
device = torch.device("cpu")

os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.empty_cache()


def get_predictions(
    data: CryptoNewsDataset,
    tokenizer,
    model: AutoModelForSeq2SeqLM,
    max_summarization_len: int,
    metric_name: str,
):
    # required because model.generate() doesn't prepend a batch_size dim
    data = [
        {
            "input_ids": sample["input_ids"].reshape(1, sample["input_ids"].size()[0]),
            "label_ids": sample["label_ids"].reshape(1, sample["label_ids"].size()[0]),
        }
        for sample in data
    ]

    output_summaries = []
    metric_values = []
    for i in tqdm(range(len(data))):
        outputs = model.generate(
            data[i]["input_ids"].to(device),
            max_length=max_summarization_len,
            min_length=10,
            no_repeat_ngram_size=2,
            num_beams=4,
            early_stopping=True,
        )
        output_summary = tokenizer.decode(outputs[0])
        output_summary = process_output(output_summary)
        output_summaries.append(output_summary)
        decoded_label = tokenizer.decode(data[i]["label_ids"][0])
        metric_val = -1
        if metric_name == "rouge":
            metric_val = calculate_rouge([output_summary], [decoded_label])
        elif metric_name == "bleu":
            metric_val = calculate_bleu(output_summary, decoded_label)
        metric_values.append(metric_val)

    return output_summaries, metric_values


def process_output(output_summary):
    output_summary = output_summary.replace("<pad>", "")
    output_summary = output_summary.replace("<unk>", "")
    output_summary = output_summary.replace("</s>", "")
    output_summary = output_summary.strip()
    return output_summary


def fine_tune(
    tokenizer,
    model: AutoModelForSeq2SeqLM,
    max_summarization_len: int,
    metric_name: str,
    model_name: str,
):
    print("Preparing tr data...")
    start = time.time()
    prepared_tr_data = data_operations.prepare_data(tr_data, tokenizer, max_summarization_len)
    end = time.time()
    print(f"Finished tr data preparation in {round(end - start, 2)}s")
    model.to(device)
    batch_size = 1
    training_args = TrainingArguments(
        f"{model_name}/trainer_output_dir",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=False,
        save_total_limit=1,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=prepared_tr_data,
    )
    trainer.compute_metrics = build_compute_metrics_fn(tokenizer, metric_name)
    trainer.train()


def write_results_to_file(
    metric_name: str, metrics_to_write: Dict[str, float], num_tr_samples: int, do_fine_tuning: bool
):
    lines = [f"{model_name}: {avg_dist}\n" for model_name, avg_dist in metrics_to_write.items()]
    print("--")
    print(lines)
    print("--")
    file_name = (
        f"avg_{metric_name}_per_model_{num_tr_samples}.txt"
        if not do_fine_tuning
        else f"fine_tuned_avg_{metric_name}_per_model_{num_tr_samples}.txt"
    )
    with open(file_name, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    huggingface_model_names = [
        "t5-base",
        "microsoft/prophetnet-large-uncased",
        "facebook/bart-base",
    ]
    num_tr_samples, num_tst_samples = 100, 100
    tr_path = "data/crypto_news_parsed_2018_validation.csv"
    tst_path = "data/crypto_news_parsed_2013-2017_train.csv"
    tr_data = data_operations.read_data(tr_path)[:num_tr_samples]
    tst_data = data_operations.read_data(tst_path)[:num_tst_samples]
    metrics_to_write = {}
    max_summarization_len = 1000
    do_fine_tuning = False
    metric_name = ["rouge", "bleu"][0]
    for model_name in huggingface_model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if do_fine_tuning:
            print(f"Fine-tuning model {model_name}")
            fine_tune(tokenizer, model, max_summarization_len, metric_name, model_name)
            print("Preparing tst data...")
        start = time.time()
        prepared_tst_data = data_operations.prepare_data(tst_data, tokenizer, max_summarization_len)
        end = time.time()
        print(f"Finished tst data preparation in {round(end - start, 2)}s")
        print(f"Getting predictions for model {model_name}...")
        summaries, metric_values = get_predictions(
            prepared_tst_data, tokenizer, model, max_summarization_len, metric_name
        )
        avg_metric_value = sum(metric_values) / len(metric_values)
        print(model_name, avg_metric_value)
        metrics_to_write[model_name] = avg_metric_value
    write_results_to_file(metric_name, metrics_to_write, num_tr_samples, do_fine_tuning)
