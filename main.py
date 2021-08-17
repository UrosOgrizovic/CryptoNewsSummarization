import time
from typing import List, Dict

import tokenizers
from Levenshtein import distance
from tqdm import tqdm
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, Trainer
import data_operations
from crypto_news_dataset import CryptoNewsDataset
import torch
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "2, 3"
import gc

gc.collect()

torch.cuda.empty_cache()
device = torch.device("cpu")


# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     torch.cuda.empty_cache()


def get_predictions(model_name: str, data: CryptoNewsDataset):
    # required because model.generate() doesn't prepend a batch_size dim
    data = [
        {
            "input_ids": sample["input_ids"].reshape(1, sample["input_ids"].size()[0]),
            "label_ids": sample["label_ids"].reshape(1, sample["label_ids"].size()[0]),
        }
        for sample in data
    ]
    # Initialize the HuggingFace summarization pipeline
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    output_summaries = []
    distances = []
    for i in tqdm(range(len(data))):
        outputs = model.generate(
            data[i]["input_ids"].to(device),
            max_length=40,
            min_length=4,
            no_repeat_ngram_size=2,
            num_beams=4,
            early_stopping=True,
        )
        output_summary = tokenizer.decode(outputs[0])
        output_summary = process_output(output_summary)
        output_summaries.append(output_summary)
        decoded_label = tokenizer.decode(data[i]["label_ids"][0])
        lev_distance = distance(output_summary, decoded_label)
        distances.append(lev_distance)

    return output_summaries, distances


def process_output(output_summary):
    output_summary = output_summary.replace("<pad>", "")
    output_summary = output_summary.replace("<unk>", "")
    output_summary = output_summary.replace("</s>", "")
    output_summary = output_summary.strip()
    return output_summary


def default_pipeline(data):
    summarizer = pipeline("summarization")
    summarized_text = summarizer(data, min_length=75, max_length=3000)
    return summarized_text


def fine_tune(model_name, data):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.to(device)
    # loss = model(input_ids=data.encodings, labels=data.labels)
    # print(loss)
    # TODO: implement model fine-tuning

    training_args = TrainingArguments(
        "trainer_output_dir",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=data,
        # eval_dataset=data[10:20],
    )
    trainer.train()


def calculate_rouge(actual, predicted):
    # TODO: implement
    pass


def write_results_to_file(
    total_distances: Dict[str, float], num_tr_samples: int, do_fine_tuning: bool
):
    lines = [f"{model_name}: {avg_dist}\n" for model_name, avg_dist in total_distances.items()]
    print("--")
    print(lines)
    print("--")
    file_name = (
        f"avg_distance_per_model_{num_tr_samples}.txt"
        if not do_fine_tuning
        else f"fine_tuned_avg_distance_per_model_{num_tr_samples}.txt"
    )
    with open(file_name, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    huggingface_model_names = [
        # "t5-base",
        # "microsoft/prophetnet-large-uncased",
        "facebook/bart-base",
    ]
    num_tr_samples, num_tst_samples = 10000, 10000
    tr_path = "data/crypto_news_parsed_2018_validation.csv"
    tst_path = "data/crypto_news_parsed_2013-2017_train.csv"
    tr_data = data_operations.read_data(tr_path)[:num_tr_samples]
    tst_data = data_operations.read_data(tst_path)[:num_tst_samples]
    total_distances = {}
    max_len = 20
    do_fine_tuning = True
    for model_name in huggingface_model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if do_fine_tuning:
            print("Preparing tr data...")
            start = time.time()
            prepared_tr_data = data_operations.prepare_data(tr_data, tokenizer, max_len)
            end = time.time()
            print(f"Finished tr data preparation in {round(end - start, 2)}s")
            print(f"Fine-tuning model {model_name}")
            model = fine_tune(model_name, prepared_tr_data)
        print("Preparing tst data...")
        start = time.time()
        prepared_tst_data = data_operations.prepare_data(tst_data, tokenizer, max_len)
        end = time.time()
        print(f"Finished tst data preparation in {round(end - start, 2)}s")
        print(f"Getting predictions for model {model_name}...")
        summaries, distances = get_predictions(model_name, prepared_tst_data)
        avg_dist = sum(distances) / len(distances)
        print(model_name, avg_dist)
        total_distances[model_name] = avg_dist
    # write_results_to_file(total_distances, num_tr_samples, do_fine_tuning)
