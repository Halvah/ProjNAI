from transformers import pipeline, BertTokenizer, BertForSequenceClassification
import torch

model_path = "../models/fake-news-bert-detect/outputs/bert-fine-tuned/model"
tokenizer_path = "../models/fake-news-bert-detect/outputs/bert-fine-tuned/tokenizer"

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

label2id = {
    "neutral": 0,
    "political bias": 1,
    "sensationalism": 2,
    "factual": 3
}

id2label = {v: k for k, v in label2id.items()}

bias_pipeline = pipeline("zero-shot-classification", model="mediabiasgroup/magpie-babe-ft-xlm", label2id=label2id, id2label=id2label)

def predict_fake_news(news_text: str):

    inputs = tokenizer(news_text, truncation=True, padding="max_length", max_length=512, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    if predicted_class == 0:
        return "Fake News"
    else:
        return "Real News"


def analyze_bias(news_text: str):
    """Analyze potential bias in the news content using the HuggingFace 'magpie-babe-ft-xlm' model."""
    labels = ["political bias", "sensationalism", "neutral", "factual"]

    result = bias_pipeline(news_text, candidate_labels=labels)

    sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])

    return sorted_results


def process_news(news_text: str):
    """Process news text to detect fake news and analyze media bias."""
    fake_news_result = predict_fake_news(news_text)

    bias_result = analyze_bias(news_text)

    return {
        "Fake News Detection": fake_news_result,
        "Bias Analysis": bias_result
    }

if __name__ == "__main__":
    news_input = input("Enter news text for fake news detection: ")
    results = process_news(news_input)

    print("\nResults:")
    print(f"Fake News Detection: {results['Fake News Detection']}")

    print("Bias Analysis:")
    for label, score in results["Bias Analysis"]:
        print(f"{label}: {score:.6f}")