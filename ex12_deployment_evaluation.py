#!/usr/bin/env python3
"""
Ex. No: 12 - DEPLOYMENT AND EVALUATION OF A GENERATIVE AI APPLICATION
             USING CLOUD-BASED APIs AND AI FRAMEWORKS

AIM
    To deploy a Generative AI text-generation application as a web service
    using the Gradio framework, and to evaluate its output quality using
    standard NLP evaluation metrics.

REQUIREMENTS
    pip install gradio transformers torch evaluate rouge_score
"""

import evaluate
import gradio as gr
from transformers import pipeline

# ---------- 1. Build and Deploy the App ----------
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


def summarize_text(input_text):
    result = summarizer(input_text, max_length=45, min_length=15, do_sample=False)
    return result[0]["summary_text"]


def evaluate_output():
    # ---------- 2. Evaluate Generated Output ----------
    rouge = evaluate.load("rouge")

    generated_summaries = [
        "AI models generate new content such as text and images.",
    ]
    reference_summaries = [
        "Generative AI models are capable of producing new content including text and images.",
    ]

    scores = rouge.compute(
        predictions=generated_summaries, references=reference_summaries
    )
    print("ROUGE Evaluation Scores:", scores)


def main():
    demo = gr.Interface(
        fn=summarize_text,
        inputs=gr.Textbox(lines=8, label="Enter text to summarize"),
        outputs=gr.Textbox(label="Generated Summary"),
        title="GenAI Text Summarizer",
        description="A cloud-deployable Generative AI summarization app built with Gradio.",
    )

    # Score the sample outputs before handing control to the blocking server.
    evaluate_output()

    demo.launch(share=True)  # share=True generates a public cloud URL


if __name__ == "__main__":
    main()
