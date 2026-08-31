#!/usr/bin/env python3
"""
Ex. No: 4 - TEXT SUMMARIZATION AND QUESTION-ANSWERING SYSTEM USING
            LARGE LANGUAGE MODELS

AIM
    To develop a text summarization system and a question-answering system
    using pre-trained Large Language Models (BART and DistilBERT).

REQUIREMENTS
    pip install transformers torch
"""

from transformers import pipeline

ARTICLE = """Generative AI refers to a class of artificial intelligence models capable of
producing new content such as text, images, audio, and video. Large Language Models (LLMs)
such as GPT and LLaMA are trained on massive text corpora and can perform a wide range of
natural language tasks including translation, summarization, and question answering. These
models are increasingly being deployed in industry applications ranging from customer support
to software development, transforming how humans interact with machines."""


def main():
    # ---------- Text Summarization ----------
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(ARTICLE, max_length=45, min_length=20, do_sample=False)
    print("Summary:\n", summary[0]["summary_text"])

    # ---------- Question Answering ----------
    qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

    context = ARTICLE
    question = "What are Large Language Models trained on?"
    answer = qa(question=question, context=context)
    print("\nQuestion:", question)
    print("Answer:", answer["answer"], "| Confidence:", round(answer["score"], 3))


if __name__ == "__main__":
    main()
