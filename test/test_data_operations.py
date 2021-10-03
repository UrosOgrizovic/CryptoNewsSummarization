import unittest
from typing import List, Dict

import tokenizers
from transformers import AutoTokenizer

import data_operations
from crypto_news import CryptoNews
from crypto_news_dataset import CryptoNewsDataset


class TestDataOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.tr_data_path = "../data/crypto_news_parsed_2013-2017_train.csv"
        self.tst_data_path = "../data/crypto_news_parsed_2018_validation.csv"

    def test_read_tr_data(self):
        tr_data = data_operations.read_data(self.tr_data_path, num_rows=10)
        self.assertIsInstance(tr_data[0], CryptoNews)

    def test_prepare_data(self):
        tst_data = data_operations.read_data(self.tst_data_path, num_rows=10)
        tokenizer = AutoTokenizer.from_pretrained("t5-base")
        max_len = 10
        prepared_tst_data = data_operations.prepare_data(tst_data, tokenizer, max_len)
        self.assertIsInstance(prepared_tst_data, CryptoNewsDataset)

    def test_text_preprocessing(self):
        txt = "  <s>Bitcoin Price Update: Will China Lead us Down?</s>\r\n  "
        expected = "bitcoin price update: will china lead us down?"
        self.assertEqual(expected, data_operations.text_preprocessing(txt))
