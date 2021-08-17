from collections import Counter
from typing import Dict, List
import constants
from crypto_news import CryptoNews


class AccumulationUtility:
    @staticmethod
    def get_num_news_per_year(data: List[CryptoNews]) -> Dict[str, int]:
        num_news_per_year = dict(Counter([news.year for news in data]))
        num_news_per_year = {str(k): v for k, v in num_news_per_year.items()}
        return num_news_per_year

    @staticmethod
    def get_num_news_per_author(data: List[CryptoNews]) -> Dict[str, int]:
        list_of_authors = [
            news.author if news.author.strip() != "" else "Unknown author" for news in data
        ]
        return dict(Counter(list_of_authors))

    @staticmethod
    def get_num_news_per_source(data: List[CryptoNews]) -> Dict[str, int]:
        list_of_sources = [
            news.source if news.source.strip() != "" else "Unknown source" for news in data
        ]
        return dict(Counter(list_of_sources))

    @staticmethod
    def get_word_prominence(data: List[CryptoNews]) -> Dict[str, int]:
        split_titles = [news.title.split() for news in data]
        filtered_titles = []
        for title in split_titles:
            for word in title:
                word = word.strip()
                if len(word) >= 3 and word.lower() not in constants.STOP_WORDS:
                    filtered_titles.append(word)
        return dict(Counter(filtered_titles))

    @staticmethod
    def get_text_lengths(data: List[CryptoNews]) -> Dict[str, int]:
        return {str(i): len(datum.text.split()) for i, datum in enumerate(data)}
