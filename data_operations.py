import csv

from typing import List, Dict

import tokenizers
import torch

from crypto_news import CryptoNews
from crypto_news_dataset import CryptoNewsDataset
from plotting_utility import PlottingUtility

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")


def read_data(path="data/crypto_news_parsed_2013-2017_train.csv") -> List[CryptoNews]:
    data = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            data.append(CryptoNews(row=row))
    return data


def prepare_data(data: List[CryptoNews], tokenizer, max_len: int) -> CryptoNewsDataset:
    samples = []
    for datum in data:
        txt = "summarize:" + datum.text
        news_encoding = (
            tokenizer(txt, return_tensors="pt", max_length=max_len, truncation=True)
            .input_ids[0]
            .to(device)
        )
        title_encoding = (
            tokenizer(datum.title, return_tensors="pt", max_length=max_len, truncation=True)
            .input_ids[0]
            .to(device)
        )
        news_encoding = torch.nn.ConstantPad1d((0, max_len - news_encoding.size()[0]),
                                               0)(news_encoding)
        title_encoding = torch.nn.ConstantPad1d((0, max_len - title_encoding.size()[0]), 0)(
            title_encoding
        )
        samples.append({"input_ids": news_encoding, "label_ids": title_encoding})
    return CryptoNewsDataset(samples)


def text_preprocessing(texts: List[str]) -> List[str]:
    # TODO: try lowercase
    return [text.strip().replace("\r", "").replace("\n", "") for text in texts]


if __name__ == "__main__":
    data = read_data()
    # print(len(data))
    # num_news_per_source = AccumulationUtility.get_num_news_per_source(data)
    # print(num_news_per_source)
    # words_in_titles = AccumulationUtility.get_words_in_titles(data)
    # print(words_in_titles)
    plotting_utility = PlottingUtility()
    plotting_utility.run_all_plots(data, "plots")
