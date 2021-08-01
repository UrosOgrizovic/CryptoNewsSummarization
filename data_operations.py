import csv

from typing import List, Dict

import tokenizers

from crypto_news import CryptoNews
from crypto_news_dataset import CryptoNewsDataset
from plotting_utility import PlottingUtility


def read_data(path="data/crypto_news_parsed_2013-2017_train.csv") -> List[CryptoNews]:
    data = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            data.append(CryptoNews(row=row))
    return data


def prepare_data(data: List[CryptoNews], tokenizer) -> CryptoNewsDataset:
    encodings_labels = {}
    for datum in data:
        txt = "summarize:" + datum.text
        encoding = tokenizer.encode(txt, return_tensors="pt", max_length=3000, truncation=True)
        encodings_labels[encoding] = datum.title
    return CryptoNewsDataset(list(encodings_labels.keys()), list(encodings_labels.values()))


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
