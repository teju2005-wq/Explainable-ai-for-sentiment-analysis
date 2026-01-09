# Explainable-ai-for-sentiment-analysis

🧠 Explainable AI for Sentiment Analysis

This project presents a two-stage Natural Language Processing (NLP) pipeline that performs fine-grained sentiment analysis by first extracting opinion-bearing phrases and then classifying their sentiment. The approach improves interpretability compared to traditional sentence-level sentiment analysis.

📌 Project Overview

Traditional sentiment analysis often predicts sentiment at the sentence level, which lacks interpretability. This project introduces a two-stage transformer-based pipeline:

Phrase Extraction (Token Classification)
Identifies opinion-bearing phrases using transformer-based Named Entity Recognition (NER).

Sentiment Classification
Classifies the extracted phrases into Positive / Negative sentiment categories.

By separating phrase extraction from sentiment prediction, the system provides explainable and fine-grained sentiment reasoning.

🚀 Features

Transformer-based Token Classification (NER) for phrase extraction

Phrase-level Sentiment Classification

Supports multiple transformer architectures:

BERT

RoBERTa

XLM-RoBERTa

BART

6-fold Cross-Validation for robust evaluation

Performance evaluation using:

Accuracy

Precision

Recall

F1-score

Confusion Matrices

Improved interpretability over traditional sentiment analysis methods

🏗️ Architecture

Stage 1: Phrase Extraction

Tokenizes input text

Applies BIO tagging (BOC, IOC, O)

Fine-tunes transformer models for token classification

Stage 2: Sentiment Classification

Uses extracted phrases

Applies transformer-based sequence classification

Predicts sentiment polarity

🧪 Models Used

BERT-base-uncased

RoBERTa-base

XLM-RoBERTa-base

BART-base

📊 Results Summary

Phrase Extraction

Accuracy: ~82%

Macro F1-score: ~72%

Sentiment Classification

Accuracy: ~99%

F1-score: ~99%

XLM-RoBERTa achieved the best overall performance across tasks.

🛠️ Tech Stack

Programming Language: Python

Frameworks & Libraries:

PyTorch

HuggingFace Transformers & Datasets

Scikit-learn

Pandas, NumPy

Matplotlib

Environment: Google Colab (GPU – NVIDIA Tesla T4)

📂 Project Structure
├── bert.ipynb
├── robertabase.ipynb
├── xlmroberta.ipynb
├── bart.ipynb
├── Explainable_AI_Report.pdf
├── Conference_Paper.pdf
└── README.md

▶️ How to Run

Clone the repository

git clone https://github.com/your-username/explainable-ai-sentiment-analysis.git
cd explainable-ai-sentiment-analysis


Open notebooks in Google Colab or local Jupyter environment.

Install required dependencies

pip install transformers datasets torch scikit-learn pandas numpy matplotlib


Run notebooks in the following order:

Token Classification notebooks

Sentiment Classification notebooks

🎯 Applications

Customer feedback analysis

Product reviews interpretation

Opinion mining

Explainable NLP systems

Social media sentiment analysis
