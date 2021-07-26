from typing import List

import matplotlib.pyplot as plt
from crypto_news import CryptoNews
from plot_param import PlotParam


class PlottingUtility:
    def __init__(self, dir_path="plots", input_data: List[CryptoNews] = None):
        self.dir_path = dir_path
        self.input_data = input_data

    def run_all_plots(self, plot_params: List[PlotParam]):

        for plot_param in plot_params:
            plot_param = self.prepare_plot(plot_param)
            self.perform_plot(plot_param)

    def prepare_plot(self, plot_param: PlotParam):
        dictionary = plot_param.accumulation_func(self.input_data)
        sorted_tuples = sorted(dictionary.items(), key=lambda x: x[1])[-5:]
        dictionary = {k: v for k, v in sorted_tuples}
        plot_param.data_to_plot = dictionary
        return plot_param

    def perform_plot(self, plot_param: PlotParam):
        plt.bar(plot_param.data_to_plot.keys(), plot_param.data_to_plot.values(), width=0.4)
        plt.xlabel(plot_param.x_label, labelpad=10)
        plt.ylabel(plot_param.y_label)
        plt.locator_params(integer=True)  # so as not to consider ints as floats
        plt.savefig(f"{self.dir_path}/{plot_param.file_name}")
        plt.clf()  # flushing is necessary
