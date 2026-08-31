#!/usr/bin/env python3
"""
Ex. No: 5 - SENTIMENT ANALYSIS AND DOCUMENT CLASSIFICATION USING
            FOUNDATION MODELS

AIM
    To perform sentiment analysis and multi-class document classification
    using pre-trained foundation models.

REQUIREMENTS
    pip install transformers torch
"""

from transformers import pipeline


def main():
    # ---------- Sentiment Analysis ----------
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    reviews = [
        "The new smartphone has an amazing camera and battery life!",
        "The delivery was late and the packaging was damaged.",
    ]

    for review in reviews:
        result = sentiment_analyzer(review)[0]
        print(f"Review: {review}\n -> {result['label']} ({round(result['score'], 3)})\n")

    # ---------- Document Classification (Zero-Shot) ----------
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    document = "The central bank raised interest rates to control rising inflation."
    candidate_labels = ["Politics", "Economy", "Sports", "Technology"]

    classification = classifier(document, candidate_labels)
    print("Document:", document)
    for label, score in zip(classification["labels"], classification["scores"]):
        print(f"{label}: {round(score, 3)}")


if __name__ == "__main__":
    main()
