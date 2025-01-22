from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import Annotated

fake_news_model_path = "../models/fake-news-bert-detect/outputs/bert-fine-tuned/model"
fake_news_tokenizer = AutoTokenizer.from_pretrained("../models/fake-news-bert-detect/outputs/bert-fine-tuned/tokenizer")
fake_news_model = AutoModelForSequenceClassification.from_pretrained(fake_news_model_path)
fake_news_pipeline = pipeline("text-classification", model=fake_news_model, tokenizer=fake_news_tokenizer)

bias_model_path = "mediabiasgroup/magpie-babe-ft-xlm"
bias_tokenizer = AutoTokenizer.from_pretrained(bias_model_path)
bias_model = AutoModelForSequenceClassification.from_pretrained(bias_model_path)
bias_pipeline = pipeline("zero-shot-classification", model=bias_model, tokenizer=bias_tokenizer)

def chunk_text(text, max_length=512):
    """Splits the input text into chunks that fit within the model's token limit."""
    tokens = fake_news_tokenizer.encode(text, truncation=True, max_length=max_length)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    return [fake_news_tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

def detect_fake_news(text: str):
    """Detect whether a given news content is fake or real using a HuggingFace model."""
    chunks = chunk_text(text)

    results = []
    for chunk in chunks:
        encoded_input = fake_news_tokenizer(chunk, truncation=True, padding=True, max_length=512, return_tensors="pt")
        result = fake_news_pipeline(encoded_input)
        results.append(result)

    return f"Fake News Classification Results: {results}"

def analyze_bias(text: str):
    """Evaluate potential bias in text using zero-shot classification."""
    labels = ["political bias", "sensationalism", "neutral", "factual"]
    result = bias_pipeline(text, candidate_labels=labels)
    sorted_results = sorted(zip(result["labels"], result["scores"]), key=lambda x: -x[1])
    return f"Bias Analysis: {sorted_results}"

def process_news_chain(user_input):
    print("Step 1: Detecting Fake News...")
    fake_news_result = detect_fake_news(user_input)
    print(fake_news_result)

    print("Step 2: Analyzing Bias...")
    bias_result = analyze_bias(user_input)
    print(bias_result)

    return {"fake_news_result": fake_news_result, "bias_result": bias_result}

if __name__ == "__main__":
    print("Welcome to the Fake News and Bias Analysis System!")
    while True:
        print("\nEnter news content for analysis (type 'exit' to quit):")
        user_input = input()
        if user_input.lower() == "exit":
            break
        print("\nProcessing your input...")
        results = process_news_chain(user_input)
        print("\nResults:")
        print(f"Fake News Detection: {results['fake_news_result']}")
        print(f"Bias Analysis: {results['bias_result']}")
