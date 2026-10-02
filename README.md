# Sentiment Analysis on Twitter Data

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=flat&logo=scikitlearn&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Apple%20M1%20GPU-black?style=flat&logo=apple&logoColor=white)
![Dataset](https://img.shields.io/badge/Dataset-Sentiment140%20(100K%20tweets)-1DA1F2?style=flat&logo=twitter&logoColor=white)
![CV](https://img.shields.io/badge/Evaluation-10--Fold%20Stratified%20CV-green?style=flat)

> **Context:** Built as a capstone project for my Master's in AI and Software Engineering. Included here to show depth in applied ML/NLP alongside my data analytics work.

---

## Overview

End-to-end NLP pipeline that compares **three fundamentally different machine learning approaches** for binary sentiment classification on 100,000 tweets (50K positive, 50K negative) from the Sentiment140 dataset.

| Approach | Models |
|---|---|
| Instance-based learning | KNN (k=5, cosine similarity + LSA) |
| Linear classifier | SVM (LinearSVC, C=1.0) |
| Deep sequence model | Bidirectional LSTM (trainable & Word2Vec embeddings) |

Each model is paired with multiple vectorization strategies and evaluated with rigorous **10-fold stratified cross-validation** — giving stable, unbiased performance estimates.

---

## Results

> Full results with fold-level breakdowns: [`report.md`](report.md)

| Rank | Model | Vectorizer | Accuracy | Std |
|:---:|---|---|:---:|:---:|
| 🥇 | **LSTM** | Trainable Embeddings | **77.24%** | ±0.62% |
| 🥈 | SVM | Hashing Vectorizer | 75.90% | ±0.33% |
| 🥉 | SVM | Count Vectorizer | 75.10% | ±0.33% |
| 4 | LSTM | Word2Vec Embeddings | 75.62% | ±0.23% |
| 5 | SVM | Word2Vec | 72.95% | ±0.30% |
| 6 | KNN | Word2Vec | 68.91% | ±0.33% |
| 7 | KNN | Hashing Vectorizer | 68.29% | ±0.41% |
| 8 | KNN | Count Vectorizer | 68.21% | ±0.58% |

### Key Finding
The **Bidirectional LSTM with trainable embeddings achieves the highest accuracy (77.24%)**, learning task-specific word representations from scratch. Interestingly, SVM with hashing vectorizer (75.90%) comes close at a fraction of the training time — demonstrating that classical methods remain competitive on well-preprocessed text data.

---

## Architecture

```
Raw Tweet
   │
   ▼
Text Cleaning  →  lowercase, remove URLs/@mentions/#hashtags/emojis
   │
   ▼
Tokenization   →  NLTK word_tokenize
   │
   ▼
Lemmatization  →  WordNetLemmatizer  (running → run)
   │
   ▼
Stop-word Removal  →  negations preserved (not, never, don't, ...)
   │
   ├──────────────────────────────────┐──────────────────────────────────────┐
   ▼                                  ▼                                      ▼
Count / Hash Vectorizer          Word2Vec (avg)                    Keras Tokenizer
(BoW + bigrams, 50K features)   (100-dim dense, window=5)         + Embedding layer
   │                                  │                                      │
   ├── KNN (k=5, cosine, LSA=150)     ├── KNN / SVM                         ▼
   └── SVM (LinearSVC, C=1.0)         └── SVM                     Bidirectional LSTM
                                                                   (128 units × 2 dir)
                                                                        │
                                                                   Dense(64, ReLU)
                                                                   Dropout(0.4)
                                                                   Dense(1, Sigmoid)
   │
   ▼
10-Fold Stratified CV  →  Accuracy · Precision · Recall (weighted)
   │
   ▼
Excel Report + 5 PNG Plots
```

---

## LSTM Architecture Detail

```
Input sequence (max 50 tokens)
        │
   Embedding(30001 × 128, glorot_uniform)   ← 3.84M trainable params
        │
   SpatialDropout1D(0.3)
        │
   Bidirectional LSTM(128)                  ← reads left→right AND right→left
        │                                      output: 256-dim (128 × 2 directions)
   Dense(64, ReLU)
        │
   Dropout(0.4)
        │
   Dense(1, Sigmoid)                        ← probability of positive sentiment
```

**Training config:** Adam (lr=1e-3, clipnorm=1.0) · EarlyStopping(patience=5) · ReduceLROnPlateau(factor=0.5, patience=2)

---

## Plots

| | |
|---|---|
| ![Metrics Comparison](results/plot1_metrics_comparison.png) | ![Heatmap](results/plot2_performance_heatmap.png) |
| Accuracy / Precision / Recall across all configurations | Performance heatmap — red=low, green=high |
| ![Boxplots](results/plot3_fold_boxplots.png) | ![Rankings](results/plot4_model_ranking.png) |
| Fold-level score distributions (stability across 10 folds) | All configurations ranked by accuracy |

---

## Tech Stack

| Component | Library / Tool |
|---|---|
| Deep Learning | TensorFlow 2.16 + Keras + tensorflow-metal (Apple M1 GPU) |
| Classical ML | scikit-learn (KNN, LinearSVC, StratifiedKFold, TruncatedSVD) |
| Word Embeddings | Gensim Word2Vec |
| NLP Preprocessing | NLTK (tokenize, lemmatize, stopwords) + emoji |
| Data | pandas, numpy |
| Visualisation | matplotlib, seaborn |
| Output | openpyxl (Excel with per-fold sheets) |

---

## Project Structure

```
├── README.md
├── report.md
├── requirements.txt
├── src/
│   ├── sentiment_analysis.py   ← full pipeline (KNN + SVM + LSTM, 10-fold CV, Excel + plots)
│   └── lstm_trainable.py       ← standalone LSTM re-run with improved config
└── results/
    ├── plot1_metrics_comparison.png
    ├── plot2_performance_heatmap.png
    ├── plot3_fold_boxplots.png
    ├── plot4_model_ranking.png
    └── plot5_radar_charts.png
```

---

## Reproducing the Results

```bash
# 1. Create virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (downloads dataset automatically if not present)
python src/sentiment_analysis.py
```

> **Apple M1 note:** Install `tensorflow-macos` + `tensorflow-metal` instead of standard `tensorflow` for GPU acceleration. Do **not** use `recurrent_dropout` in LSTM layers — it disables the Metal GPU kernel.

---

## What I Learned

- **Negation-aware preprocessing** is critical: removing "not", "don't", "never" from stop-word lists dramatically changes sentiment classification.
- **SVM is a strong baseline** for NLP — LinearSVC with bigrams (75.90%) nearly matches a deep LSTM at ~1% of the training time.
- **Trainable embeddings need time**: with only 5 epochs the LSTM was stuck at 50% accuracy. Increasing to 20 epochs with early stopping and ReduceLROnPlateau unlocked real learning.
- **Data leakage prevention**: fitting CountVectorizer and Keras Tokenizer strictly on the training fold inside each CV split is essential for honest evaluation.
- **Hardware-specific constraints matter**: `recurrent_dropout` in TensorFlow silently falls back from GPU to CPU on Apple M1, causing 7× slower training per fold.

---

## Author
**Elnaz Khodadadi** — Data Analyst | Excel · SQL · Power BI · Python
GitHub: [@ekhodada](https://github.com/ekhodada) · LinkedIn: [elnaz-khodadadi](https://www.linkedin.com/in/elnaz-khodadadi-547868a6/)
