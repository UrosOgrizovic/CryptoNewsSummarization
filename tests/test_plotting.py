import os
import unittest
import inspect

import data_operations
from accumulation_utility import AccumulationUtility
from plotting_utility import PlottingUtility
from plot_param import PlotParam

TR_DATA_PATH = "../data/crypto_news_parsed_2013-2017_train.csv"


class TestPlotting(unittest.TestCase):
    def setUp(self) -> None:
        self.data = data_operations.read_data(TR_DATA_PATH)
        self.dir_path = "../plots"
        self.plot_params = [
            PlotParam(
                "year", "# news", "num_news_per_year.png", AccumulationUtility.get_num_news_per_year
            ),
            PlotParam(
                "author",
                "# news",
                "num_news_per_author.png",
                AccumulationUtility.get_num_news_per_author,
            ),
            PlotParam(
                "source",
                "# news",
                "num_news_per_source.png",
                AccumulationUtility.get_num_news_per_source,
            ),
            PlotParam(
                "word",
                "# times used",
                "word_prominence.png",
                AccumulationUtility.get_words_in_titles,
            ),
            PlotParam(
                "id",
                "text length in words",
                "text_lengths.png",
                AccumulationUtility.get_num_words_in_texts,
            ),
        ]

    def test_all_plots(self) -> None:
        PlottingUtility(self.dir_path, self.data).run_all_plots(self.plot_params)
        num_plots = len(os.listdir(self.dir_path))
        self.assertEqual(len(self.plot_params), num_plots)


if __name__ == "__main__":
    unittest.main()
