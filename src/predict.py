from transformers import pipeline
from predict_all import get_all_predictions

label2id = {
    "neutral": 0,
    "political bias": 1,
    "sensationalism": 2,
    "factual": 3
}

id2label = {v: k for k, v in label2id.items()}

bias_pipeline = pipeline("zero-shot-classification", model="mediabiasgroup/magpie-babe-ft-xlm", label2id=label2id, id2label=id2label)

def analyze_bias(news_text: str):
    """Analyze potential bias in the news content using the HuggingFace 'magpie-babe-ft-xlm' model."""
    labels = ["political bias", "sensationalism", "neutral", "factual"]

    result = bias_pipeline(news_text, candidate_labels=labels)

    sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])

    return sorted_results

def predict(text: str):
    all_predictions = get_all_predictions(text)

    bias_result = analyze_bias(text)

    return {
        "predictions": all_predictions,
        "bias_analysis": bias_result
    }

def process_news(news_text: str):
    """Process news text to detect fake news and analyze media bias."""
    fake_news_result = predict(news_text)

    bias_result = analyze_bias(news_text)

    return {
        "Fake News Detection": fake_news_result,
        "Bias Analysis": bias_result
    }

if __name__ == "__main__":
    news_input = input("Enter news text for fake news detection: ")
    results = process_news(news_input)

    print("Model Predictions:")
    print(f"BERT: {results['Fake News Detection']['predictions']['bert']}")
    print(f"ALBERT: {results['Fake News Detection']['predictions']['albert']}")
    print(f"RoBERTa: {results['Fake News Detection']['predictions']['roberta']}")
    print("\nBias Analysis:")
    bias_analysis = results['Fake News Detection']["bias_analysis"]

    print("Bias Analysis:")
    for label, score in bias_analysis:
        print(f"- {label}: {score:.2%}")