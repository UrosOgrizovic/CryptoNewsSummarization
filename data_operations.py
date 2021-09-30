import csv
import itertools
import re

from typing import List, Dict

import tokenizers
import torch

from crypto_news import CryptoNews
from crypto_news_dataset import CryptoNewsDataset

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def read_data(
    path="data/crypto_news_parsed_2013-2017_train.csv", num_rows=100000
) -> List[CryptoNews]:
    data = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in itertools.islice(reader, num_rows):
            data.append(CryptoNews(row[0], row[1], row[2], row[3], int(row[4]), row[5], row[6]))
    return data


def prepare_data(data: List[CryptoNews], tokenizer, max_len: int) -> CryptoNewsDataset:
    samples = []
    for datum in data:
        datum.text = text_preprocessing(datum.text)
        datum.title = text_preprocessing(datum.title)
        if len(datum.text) < 220 or (datum.text.startswith("http") and " " not in datum.text):
            continue
        txt = "summarize:" + datum.text
        news_encoding = (
            tokenizer(txt, return_tensors="pt", max_length=max_len, truncation=True)
            .input_ids[0]
            .to(device)
        )
        title_encoding = (
            tokenizer(
                datum.title,
                return_tensors="pt",
                max_length=max_len,
                truncation=True,
            )
            .input_ids[0]
            .to(device)
        )
        news_encoding = torch.nn.ConstantPad1d((0, max_len - news_encoding.size()[0]), 0)(
            news_encoding
        )
        # necessary for fine-tuning, but not for inference
        # title_encoding = torch.nn.ConstantPad1d((0, max_len - title_encoding.size()[0]), 0)(
        #     title_encoding
        # )
        samples.append({"input_ids": news_encoding, "label_ids": title_encoding})
    return CryptoNewsDataset(samples)


def text_preprocessing(text: str) -> str:
    # lowercasing and lemmatization didn't improve results
    text = text.lower()
    text = re.sub(" +", " ", text)
    text = text.strip().replace("\r", "").replace("\n", "")
    html_cleanr = re.compile("<.*?>")
    text = re.sub(html_cleanr, "", text)
    return text


if __name__ == "__main__":
    data = read_data()
    data2 = read_data("data/crypto_news_parsed_2018_validation.csv")
    # print(len(data))
    # num_news_per_source = AccumulationUtility.get_num_news_per_source(data)
    # print(num_news_per_source)
    # words_in_titles = AccumulationUtility.get_words_in_titles(data)
    # print(words_in_titles)
