#!/usr/bin/env python3
"""
LSTM with Trainable Embedding Layer — Sentiment140
===================================================
Standalone re-run of the trainable-embedding LSTM variant.

Fixes vs. original run:
  - More epochs (20) so embeddings have time to learn from scratch
  - Larger patience (5) to avoid stopping before convergence
  - Smaller batch size (256) for better gradient estimates
  - Bidirectional LSTM for richer context
  - Gradient clipping (clipnorm=1.0) to prevent exploding gradients
  - ReduceLROnPlateau to lower LR when val_loss plateaus
  - class_weight balancing (belt-and-suspenders)
"""

import os, re, sys, time, zipfile, warnings, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
import emoji

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, SpatialDropout1D,
)
from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════
CFG = {
    'SAMPLE_SIZE':        50_000,
    'RANDOM_STATE':       42,
    'DATA_FILE':          'training.1600000.processed.noemoticon.csv',
    'DATA_URL':           'http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip',
    'N_FOLDS':            10,

    # LSTM hyperparameters (improved)
    'LSTM_MAX_WORDS':     30_000,
    'LSTM_MAX_LEN':       50,
    'LSTM_EMBED_DIM':     128,
    'LSTM_UNITS':         128,          # increased from 64
    'LSTM_SPATIAL_DROP':  0.3,
    'LSTM_DENSE_DROP':    0.4,
    'LSTM_EPOCHS':        20,           # was 5 — embeddings need more time
    'LSTM_BATCH_SIZE':    128,          # smaller = more frequent updates + visible progress
    'LSTM_PATIENCE':      5,            # was 2 — give model room to recover
    'LSTM_LR':            1e-3,
    'LSTM_CLIPNORM':      1.0,          # gradient clipping

    'RESULTS_DIR':        Path('results'),
}

CFG['RESULTS_DIR'].mkdir(exist_ok=True)

# ════════════════════════════════════════════════════════════════════
#  NLTK RESOURCES
# ════════════════════════════════════════════════════════════════════
for _r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
    nltk.download(_r, quiet=True)

_NEGATIONS = {
    'no', 'not', 'never', 'neither', 'nor', 'nothing', 'nowhere',
    'nobody', 'none', 'cannot', "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "hadn't", "wouldn't", "shouldn't", "couldn't", "mightn't", "mustn't",
    "needn't", "shan't", "n't",
}
STOP_WORDS  = set(stopwords.words('english')) - _NEGATIONS
LEMMATIZER  = WordNetLemmatizer()


# ════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ════════════════════════════════════════════════════════════════════
def _dl_progress(block, block_size, total):
    pct = min(block * block_size / total * 100, 100)
    bar = '█' * int(pct / 2)
    print(f'\r  [{bar:<50}] {pct:.1f}%', end='', flush=True)


def download_dataset() -> None:
    if Path(CFG['DATA_FILE']).exists():
        print(f"  Dataset found: {CFG['DATA_FILE']}")
        return
    zip_path = 'trainingandtestdata.zip'
    if not Path(zip_path).exists():
        print('  Downloading Sentiment140 dataset (~80 MB) …')
        try:
            urllib.request.urlretrieve(CFG['DATA_URL'], zip_path, _dl_progress)
            print()
        except Exception as exc:
            print(f'\n  Download failed: {exc}')
            sys.exit(1)
    print('  Extracting …')
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall('.')
    print('  Dataset ready.')


def load_data() -> tuple:
    print('\n[1/4] Loading dataset …')
    df = pd.read_csv(
        CFG['DATA_FILE'],
        encoding='latin-1', header=None,
        names=['polarity', 'id', 'date', 'query', 'user', 'text'],
        usecols=['polarity', 'text'],
    )
    n   = CFG['SAMPLE_SIZE']
    neg = df[df['polarity'] == 0].sample(n, random_state=CFG['RANDOM_STATE'])
    pos = df[df['polarity'] == 4].sample(n, random_state=CFG['RANDOM_STATE'])
    data = (
        pd.concat([neg, pos])
        .sample(frac=1, random_state=CFG['RANDOM_STATE'])
        .reset_index(drop=True)
    )
    labels = (data['polarity'] == 4).astype(int).values
    texts  = data['text'].values
    print(f'  {len(data):,} tweets  |  {(labels==0).sum():,} neg  |  {(labels==1).sum():,} pos')
    return texts, labels


# ════════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ════════════════════════════════════════════════════════════════════
def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'&\w+;', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def _preprocess(text: str) -> list:
    tokens = word_tokenize(_clean(text))
    return [
        LEMMATIZER.lemmatize(t)
        for t in tokens
        if len(t) > 1 and t not in STOP_WORDS
    ]


def preprocess_all(texts) -> list:
    print('\n[2/4] Preprocessing tweets …')
    t0 = time.time()
    tokenized = []
    for i, text in enumerate(texts):
        tokenized.append(_preprocess(text))
        if (i + 1) % 20_000 == 0:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (len(texts) - i - 1)
            print(f'  {i+1:>7,}/{len(texts):,}  |  {elapsed:.0f}s  |  ETA {eta:.0f}s')
    print(f'  Done in {time.time()-t0:.1f}s')
    return tokenized


# ════════════════════════════════════════════════════════════════════
#  MODEL BUILDER
# ════════════════════════════════════════════════════════════════════
def build_model(vocab_size: int) -> tf.keras.Model:
    """Bidirectional LSTM with trainable embeddings."""
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=CFG['LSTM_EMBED_DIM'],
            embeddings_initializer='glorot_uniform',  # better than uniform default
        ),
        SpatialDropout1D(CFG['LSTM_SPATIAL_DROP']),
        Bidirectional(LSTM(
            CFG['LSTM_UNITS'],
            dropout=0.2,
            # recurrent_dropout disabled: forces CPU on TF-Metal, making each fold hours slow
            return_sequences=False,
        )),
        Dense(64, activation='relu'),
        Dropout(CFG['LSTM_DENSE_DROP']),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=CFG['LSTM_LR'], clipnorm=CFG['LSTM_CLIPNORM']),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ════════════════════════════════════════════════════════════════════
#  10-FOLD CV
# ════════════════════════════════════════════════════════════════════
def run_cv(tokenized: list, labels: np.ndarray) -> dict:
    print('\n[3/4] 10-fold stratified CV — LSTM + Trainable Embeddings …\n')
    skf    = StratifiedKFold(n_splits=CFG['N_FOLDS'], shuffle=True, random_state=CFG['RANDOM_STATE'])
    folds  = []
    history_all = []

    for fold, (tr, va) in enumerate(skf.split(np.zeros(len(labels)), labels), 1):
        y_tr, y_va = labels[tr], labels[va]

        train_texts = [' '.join(tokenized[i]) for i in tr]
        val_texts   = [' '.join(tokenized[i]) for i in va]

        kt = KerasTokenizer(num_words=CFG['LSTM_MAX_WORDS'], oov_token='<OOV>')
        kt.fit_on_texts(train_texts)
        vocab_size = min(CFG['LSTM_MAX_WORDS'], len(kt.word_index)) + 1

        X_tr = pad_sequences(
            kt.texts_to_sequences(train_texts),
            maxlen=CFG['LSTM_MAX_LEN'], padding='post', truncating='post',
        )
        X_va = pad_sequences(
            kt.texts_to_sequences(val_texts),
            maxlen=CFG['LSTM_MAX_LEN'], padding='post', truncating='post',
        )

        # class weights in case of imbalance (should be ~50/50 but belt-and-suspenders)
        neg_count = int((y_tr == 0).sum())
        pos_count = int((y_tr == 1).sum())
        class_weight = {0: 1.0, 1: neg_count / pos_count} if pos_count else {0: 1.0, 1: 1.0}

        model = build_model(vocab_size)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=CFG['LSTM_PATIENCE'],
                restore_best_weights=True,
                verbose=0,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=0,
            ),
        ]

        print(f'  ── Fold {fold}/10 ──')
        hist = model.fit(
            X_tr, y_tr,
            epochs=CFG['LSTM_EPOCHS'],
            batch_size=CFG['LSTM_BATCH_SIZE'],
            validation_data=(X_va, y_va),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        history_all.append(hist.history)

        y_pred = (model.predict(X_va, verbose=0).flatten() >= 0.5).astype(int)
        acc  = accuracy_score(y_va, y_pred)
        pre  = precision_score(y_va, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_va, y_pred, average='weighted', zero_division=0)
        folds.append((acc, pre, rec))

        best_epoch = len(hist.history['val_loss'])
        print(f'  Fold {fold:2d}/10  |  Acc {acc:.4f}  Pre {pre:.4f}  Rec {rec:.4f}'
              f'  |  stopped @ epoch {best_epoch}')

        tf.keras.backend.clear_session()

    accs  = [f[0] for f in folds]
    precs = [f[1] for f in folds]
    recs  = [f[2] for f in folds]
    result = {
        'Model': 'LSTM', 'Vectorizer': 'Trainable Embeddings',
        'Accuracy_mean':   float(np.mean(accs)),   'Accuracy_std':   float(np.std(accs)),
        'Precision_mean':  float(np.mean(precs)),  'Precision_std':  float(np.std(precs)),
        'Recall_mean':     float(np.mean(recs)),   'Recall_std':     float(np.std(recs)),
        'Accuracy_folds':  accs,
        'Precision_folds': precs,
        'Recall_folds':    recs,
    }

    print(f'\n  ── Summary ──')
    print(f'  Accuracy  : {result["Accuracy_mean"]:.4f} ± {result["Accuracy_std"]:.4f}')
    print(f'  Precision : {result["Precision_mean"]:.4f} ± {result["Precision_std"]:.4f}')
    print(f'  Recall    : {result["Recall_mean"]:.4f} ± {result["Recall_std"]:.4f}')

    return result, history_all


# ════════════════════════════════════════════════════════════════════
#  OUTPUT
# ════════════════════════════════════════════════════════════════════
def save_results(result: dict) -> None:
    print('\n[4/4] Saving results …')

    # Excel
    xlsx_path = CFG['RESULTS_DIR'] / 'lstm_trainable_results.xlsx'
    fold_df = pd.DataFrame({
        'Fold':      list(range(1, CFG['N_FOLDS'] + 1)),
        'Accuracy':  [round(v, 4) for v in result['Accuracy_folds']],
        'Precision': [round(v, 4) for v in result['Precision_folds']],
        'Recall':    [round(v, 4) for v in result['Recall_folds']],
    })
    stats_df = pd.DataFrame([
        {'Fold': 'Mean', 'Accuracy': round(result['Accuracy_mean'], 4),
         'Precision': round(result['Precision_mean'], 4),
         'Recall': round(result['Recall_mean'], 4)},
        {'Fold': 'Std',  'Accuracy': round(result['Accuracy_std'],  4),
         'Precision': round(result['Precision_std'],  4),
         'Recall': round(result['Recall_std'],  4)},
    ])
    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        pd.concat([fold_df, stats_df], ignore_index=True).to_excel(
            writer, sheet_name='LSTM_Trainable', index=False,
        )
        ws  = writer.sheets['LSTM_Trainable']
        hdr = Font(bold=True, color='FFFFFF', size=11)
        fill = PatternFill(fill_type='solid', fgColor='2F5496')
        for cell in ws[1]:
            cell.font = hdr; cell.fill = fill
            cell.alignment = Alignment(horizontal='center')
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = (
                max(len(str(c.value or '')) for c in col) + 4
            )
    print(f'  Saved: {xlsx_path}')

    # Plot — fold accuracy bar + summary annotation
    fig, ax = plt.subplots(figsize=(10, 5))
    folds_x = np.arange(1, CFG['N_FOLDS'] + 1)
    bars = ax.bar(folds_x, result['Accuracy_folds'], color='#5BAD75', alpha=0.85,
                  edgecolor='white', linewidth=1.2)
    ax.axhline(result['Accuracy_mean'], color='navy', linestyle='--', linewidth=1.5,
               label=f"Mean = {result['Accuracy_mean']:.4f}")
    ax.axhline(0.5, color='red', linestyle=':', linewidth=1.2, label='Random baseline (0.50)')
    for b, v in zip(bars, result['Accuracy_folds']):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(folds_x)
    ax.set_xticklabels([f'Fold {i}' for i in folds_x], rotation=30, ha='right')
    lo = max(0.0, min(result['Accuracy_folds']) - 0.05)
    ax.set_ylim(lo, min(1.0, max(result['Accuracy_folds']) + 0.06))
    ax.set_ylabel('Accuracy')
    ax.set_title(
        f'LSTM + Trainable Embeddings — 10-Fold CV Accuracy\n'
        f'Mean {result["Accuracy_mean"]:.4f} ± {result["Accuracy_std"]:.4f}  '
        f'| Precision {result["Precision_mean"]:.4f}  | Recall {result["Recall_mean"]:.4f}',
        fontweight='bold',
    )
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plot_path = CFG['RESULTS_DIR'] / 'lstm_trainable_accuracy.png'
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  Saved: {plot_path}')


# ════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t_start = time.time()
    print('═' * 60)
    print('  LSTM + Trainable Embeddings  (improved re-run)')
    print('═' * 60)

    download_dataset()
    texts, labels    = load_data()
    tokenized        = preprocess_all(texts)
    result, _history = run_cv(tokenized, labels)
    save_results(result)

    elapsed = time.time() - t_start
    print(f'\n  Total runtime: {elapsed/60:.1f} min')
    print('  Done.')
