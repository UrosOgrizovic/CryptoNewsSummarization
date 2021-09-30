from wordcloud import WordCloud

from accumulation_utility import AccumulationUtility
from data_operations import read_data, text_preprocessing


def generate_word_cloud():
    data = read_data()
    data = list(
        filter(
            lambda datum: len(datum.text) >= 220
            and len(datum.title) >= 10
            and not datum.text.startswith("http")
            and " " in datum.text,
            data,
        )
    )
    for datum in data:
        datum.text = text_preprocessing(datum.text)
        datum.title = text_preprocessing(datum.title)
        datum.text.replace(".", "").replace(",", "").replace("'", "")

    attr_name = "text"
    word_prominence = AccumulationUtility.get_word_prominence(data, attr_name)
    word_prominence = {word: freq for word, freq in word_prominence.items() if freq > 300}

    word_cloud_text = ""
    for word, freq in word_prominence.items():
        for _ in range(freq):
            word_cloud_text += word + " "
    num_words = 50
    wordcloud = WordCloud(
        max_words=num_words, background_color="white", width=640, height=480, collocations=False
    ).generate(word_cloud_text)
    for entry in wordcloud.layout_:
        (word, _), _, _, _, _ = entry
        if len(word) < 3:
            wordcloud.layout_.remove(entry)
    wordcloud.to_file(f"plots/wordcloud_{attr_name}_{num_words}_most_frequent.png")


if __name__ == "__main__":
    generate_word_cloud()
