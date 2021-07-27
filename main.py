from Levenshtein import distance
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import data_operations


def custom_pipeline(model_name, data):
    # Initialize the HuggingFace summarization pipeline
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # T5 uses a max_length of 512 so we cut the article to 512 tokens.
    # inputs = tokenizer.encode(
    #     "summarize: " + ARTICLE, return_tensors="pt", max_length=512, truncation=True
    # )
    output_summaries = []
    distances = []
    for datum in data:
        txt = "summarize:" + datum.text
        inputs = tokenizer.encode(txt, return_tensors="pt", max_length=3000, truncation=True)
        outputs = model.generate(
            inputs,
            max_length=150,
            min_length=40,
            no_repeat_ngram_size=2,
            num_beams=4,
            early_stopping=True,
        )
        output_summary = tokenizer.decode(outputs[0])
        output_summary = output_summary.replace("<pad>", "")
        output_summary = output_summary.replace("<unk>", "")
        output_summary = output_summary.replace("</s>", "")
        output_summary = output_summary.strip()
        output_summaries.append(output_summary)

        lev_distance = distance(output_summary, datum.title)
        distances.append(lev_distance)

    return output_summaries, distances


def default_pipeline(data):
    summarizer = pipeline("summarization")
    summarized_text = summarizer(data, min_length=75, max_length=3000)
    return summarized_text


def fine_tune(model, data):
    # TODO: implement model fine-tuning
    pass


def calculate_rouge(actual, predicted):
    # TODO: implement
    pass


if __name__ == "__main__":
    huggingface_model_names = [
        "t5-base",
        "microsoft/prophetnet-large-uncased",
        "facebook/bart-base",
    ]
    data = data_operations.read_data("data/crypto_news_parsed_2018_validation.csv")
    for model_name in huggingface_model_names:
        summaries, distances = custom_pipeline(model_name, data)
        print(model_name, sum(distances) / len(distances))
