import csv
import time
from typing import Dict

from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import data_operations
from crypto_news_dataset import CryptoNewsDataset
import torch
import os
from metrics import calculate_rouge, calculate_bleu

device = torch.device("cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"

# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.empty_cache()


def get_predictions(
    data: CryptoNewsDataset,
    tokenizer,
    model: AutoModelForSeq2SeqLM,
    max_input_seq_len: int,
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
    misclassified = []
    for i in tqdm(range(len(data))):
        outputs = model.generate(
            data[i]["input_ids"].to(device),
            max_length=max_input_seq_len,
            min_length=10,
            no_repeat_ngram_size=2,
            num_beams=4,
            early_stopping=True,
        )
        output_summary = tokenizer.decode(outputs[0])
        output_summary = process_text(output_summary)
        output_summaries.append(output_summary)
        decoded_label = tokenizer.decode(data[i]["label_ids"][0])
        metric_val = -1
        misclassification_threshold = 20
        if metric_name == "rouge1":
            metric_val = calculate_rouge([output_summary], [decoded_label], rouge_keys=["rouge1"])
        elif metric_name == "rouge2":
            metric_val = calculate_rouge([output_summary], [decoded_label], rouge_keys=["rouge2"])
            misclassification_threshold = 5
        elif metric_name == "bleu":
            if len(output_summary.split()) < 4 or len(decoded_label.split()) < 4:
                continue
            metric_val = calculate_bleu([output_summary], [decoded_label])
            misclassification_threshold = 1
        metric_val = list(metric_val.values())[0]
        if metric_val < misclassification_threshold:
            decoded_input = tokenizer.decode(data[i]["input_ids"][0])
            decoded_input = process_text(decoded_input)
            misclassified.append(
                f"{decoded_input}, title={decoded_label}, summary"
                f"={output_summary},"
                f" {metric_name}"
                f"={metric_val}"
            )
        metric_values.append(metric_val)
    print("num misclassified = ", len(misclassified))
    s = "\n".join(misclassified)
    with open(
        f"misclassified/{model_name.replace('/', '-')}_{metric_name}_misclassified.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(s)
    return output_summaries, metric_values


def process_text(text):
    text = text.replace("<pad>", "")
    text = text.replace("<unk>", "")
    text = text.replace("</s>", "")
    text = text.replace("<s>", "")
    text = text.strip()
    return text


def fine_tune(
    tr_data,
    model: AutoModelForSeq2SeqLM,
    model_name: str,
):
    batch_size = 64
    print(f"Finished tr data preparation in {round(end - start, 2)}s")
    model.to(device)
    output_dir_name = f"trainer_output_dir/{model_name.replace('/', '-')}"
    print(f"output dir name {output_dir_name}")
    if not os.path.exists(output_dir_name):
        os.makedirs(output_dir_name)
    training_args = TrainingArguments(
        output_dir=output_dir_name,
        overwrite_output_dir=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=False,
        save_total_limit=1,
        save_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_data,
    )
    trainer.train()


def write_results_to_file(
    metric_name: str, metrics_to_write: Dict[str, float], num_tst_samples: int
):
    lines = [f"{model_name}: {avg_dist}\n" for model_name, avg_dist in metrics_to_write.items()]
    print("--")
    print(lines)
    print("--")
    file_name = f"avg_{metric_name}_per_model_{num_tst_samples}.txt"
    with open(file_name, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    num_tr_samples, num_tst_samples = 10000, 2000
    tr_path = "data/crypto_news_parsed_2018_validation.csv"
    tst_path = "data/crypto_news_parsed_2013-2017_train.csv"
    tr_data = data_operations.read_data(tr_path)[:num_tr_samples]
    tst_data = data_operations.read_data(tst_path)[:num_tst_samples]
    print("len tr data", len(tr_data))
    print("len tst data", len(tst_data))
    metrics_to_write = {}
    max_input_seq_len = 50
    do_fine_tuning = False
    do_evaluation = True
    huggingface_model_names = ["t5-base", "facebook/bart-base"]
    metric_name = ["rouge1", "rouge2", "bleu"][0]
    for model_name in huggingface_model_names:
        model_path = f"trainer_output_dir/{model_name.replace('/', '-')}/checkpoint-200"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        model.to(device)
        if do_fine_tuning:
            print("Preparing tr data...")
            start = time.time()
            tr_data = data_operations.prepare_data(tr_data, tokenizer, max_input_seq_len)
            end = time.time()
            print(f"Fine-tuning model {model_name}")
            fine_tune(tr_data, model, model_name)
        if do_evaluation:
            print("Preparing tst data...")
            start = time.time()
            prepared_tst_data = data_operations.prepare_data(tst_data, tokenizer, max_input_seq_len)
            end = time.time()
            print(f"Finished tst data preparation in {round(end - start, 2)}s")
            print(f"Getting predictions for model {model_name}...")
            summaries, metric_values = get_predictions(
                prepared_tst_data,
                tokenizer,
                model,
                max_input_seq_len,
                metric_name,
            )
            avg_metric_value = sum(metric_values) / len(metric_values)
            print(f"performance on tst data: {model_name}, {metric_name}: {avg_metric_value}")
            metrics_to_write[model_name] = avg_metric_value
        if do_evaluation:
            write_results_to_file(metric_name, metrics_to_write, num_tst_samples)
