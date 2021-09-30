class CryptoNews:
    """
    class that represents a single piece of crypto news
    """

    def __init__(
        self,
        row: list = None,
        url="",
        title="",
        text="",
        html="",
        year=0,
        author="",
        source="",
    ):
        self.url = url
        self.title = title
        self.text = text
        self.html = html
        self.year = year
        self.author = author
        self.source = source

    def __str__(self):
        return f"{self.url}, {self.title}, {self.year}, {self.author}, {self.source}"

    def __repr__(self):
        return self.__str__()

    def print_text(self):
        return f"{self.text}"
