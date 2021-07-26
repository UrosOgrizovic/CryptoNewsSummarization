import unittest

import data_operations
from accumulation_utility import AccumulationUtility

TR_DATA_PATH = "../data/crypto_news_parsed_2013-2017_train.csv"


class TestAccumulationFunctions(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data_operations.read_data(TR_DATA_PATH)

    def test_accumulations(self):
        methods = [
            met for met in dir(AccumulationUtility) if callable(getattr(AccumulationUtility, met))
        ]
        accumulation_methods = [met for met in methods if met.startswith("get_num")]
        for method in accumulation_methods:
            method_to_call = getattr(AccumulationUtility, method)
            dictionary = method_to_call(self.data)
            self.assertEqual(len(self.data), sum(dictionary.values()))


if __name__ == "__main__":
    unittest.main()
