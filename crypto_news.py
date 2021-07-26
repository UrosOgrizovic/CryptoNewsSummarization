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
        self.url = row[0] or url
        self.title = row[1] or title
        self.text = row[2] or text
        self.html = row[3] or html
        self.year = int(row[4]) or year
        self.author = row[5] or author
        self.source = row[6] or source

    def __str__(self):
        return f"{self.url}, {self.title}, {self.year}, {self.author}, {self.source}"

    def __repr__(self):
        return self.__str__()

    def print_text(self):
        return f"{self.text}"
