from typing import Dict, List, Callable

from crypto_news import CryptoNews


class PlotParam:
    def __init__(
        self,
        x_label="",
        y_label="",
        file_name="",
        accumulation_func: Callable[[List[CryptoNews]], Dict[str, int]] = None,
        data_to_plot: Dict[str, int] = None,
    ):
        self.x_label = x_label
        self.y_label = y_label
        self.file_name = file_name
        self.accumulation_func = accumulation_func
        self.data_to_plot = data_to_plot
