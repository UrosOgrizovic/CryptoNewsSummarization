import csv

from typing import List

from crypto_news import CryptoNews
from plotting_utility import PlottingUtility


def read_data(path="data/crypto_news_parsed_2013-2017_train.csv") -> List[CryptoNews]:
    data = []
    with open(path, newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            data.append(CryptoNews(row=row))
    return data


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
