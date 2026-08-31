# CS4V48 - GenAI and LLM Laboratory

Executable Python programs for the twelve experiments in the CS4V48 GenAI & LLM
lab manual. Each experiment is a standalone script that can be run directly.

## Experiments

| Ex. No. | Experiment | File |
|---|---|---|
| 1 | Text Generation Using Pre-Trained Foundation Models | `ex01_text_generation.py` |
| 2 | Prompt Engineering Techniques for Content Generation, Reasoning and Task Automation | `ex02_prompt_engineering.py` |
| 3 | Conversational AI Chatbot Using Transformer-Based Language Models | `ex03_conversational_chatbot.py` |
| 4 | Text Summarization and Question-Answering System Using Large Language Models | `ex04_summarization_qa.py` |
| 5 | Sentiment Analysis and Document Classification Using Foundation Models | `ex05_sentiment_classification.py` |
| 6 | Retrieval-Augmented Generation (RAG) System Using Vector Databases | `ex06_rag_vector_db.py` |
| 7 | AI-Powered Code Generation and Debugging Assistant | `ex07_code_generation_debugging.py` |
| 8 | Image Generation Application Using Diffusion Models | `ex08_image_generation_diffusion.py` |
| 9 | Multimodal AI Application Integrating Text and Image Inputs | `ex09_multimodal_text_image.py` |
| 10 | Fine-Tuning a Pre-Trained Language Model for a Domain-Specific Application | `ex10_fine_tuning_distilbert.py` |
| 11 | AI-Based Content Generation System for Text, Image and Multimedia Applications | `ex11_multimedia_content_generation.py` |
| 12 | Deployment and Evaluation of a Generative AI Application Using Cloud-Based APIs and AI Frameworks | `ex12_deployment_evaluation.py` |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Models are downloaded from the Hugging Face Hub on first run, so the initial
execution of each script needs an internet connection.

## Running an experiment

```bash
./ex01_text_generation.py
# or
python3 ex01_text_generation.py
```

## Notes

- **Ex. 3** is interactive — it prompts for user input over five dialogue turns.
  Type `quit` to exit early.
- **Ex. 8 and 11** use Stable Diffusion. They select CUDA when a GPU is
  available and fall back to CPU otherwise; CPU generation is significantly
  slower. Google Colab with a GPU runtime is recommended.
- **Ex. 10** fine-tunes DistilBERT on 2000 IMDB samples. A GPU is recommended.
- **Ex. 12** launches a Gradio web app with `share=True`, which creates a
  public URL. ROUGE scores are printed before the server starts.
