# 📊 N-Gram Language Model

<div align="center">

**A Statistical Approach to Natural Language Generation**

Built with NLTK • Brown Corpus • Google Gemini

---

[Overview](#overview) • [Architecture](#architecture) • [Mathematical Foundation](#mathematical-foundation) • [Pipeline](#pipeline)

</div>

---

## Overview

This project implements a **trigram language model** using maximum likelihood estimation with Laplace smoothing. The model learns probabilistic word sequences from the Brown Corpus and generates novel text through weighted sampling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA PREPROCESSING                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Raw Corpus ──► Tokenization ──► Lowercasing ──► BOS/EOS Tagging          │
│                                                                             │
│   "The cat sat" ──► ["the", "cat", "sat"] ──► ["<s>", "<s>", "the", ...]   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MODEL CONSTRUCTION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                    Conditional Frequency Distribution                       │
│                                                                             │
│        ┌──────────────────┐      ┌─────────────────────────────┐           │
│        │   Context (w₁,w₂) │ ──► │  Target Distribution P(w₃)  │           │
│        └──────────────────┘      └─────────────────────────────┘           │
│                                                                             │
│        ("i", "am")         ──►   {"happy": 12, "not": 8, ...}              │
│        ("you", "are")      ──►   {"the": 15, "a": 10, ...}                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GENERATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Prompt ──► Context Lookup ──► Weighted Sampling ──► Token Append ──►Loop │
│                                                                             │
│   "I am" ──► P(w|"i","am") ──► sample("happy") ──► "I am happy" ──► ...    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              EVALUATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│          Sentence ──► Probability ──► Perplexity ──► Model Confidence      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Foundation

### Trigram Probability

The probability of word $w_3$ given the two preceding words $w_1$ and $w_2$ is estimated using **Maximum Likelihood Estimation (MLE)**:

$$P(w_3 \mid w_1, w_2) = \frac{C(w_1, w_2, w_3)}{C(w_1, w_2)}$$

Where:
- $C(w_1, w_2, w_3)$ = count of trigram occurrences
- $C(w_1, w_2)$ = count of bigram context occurrences

---

### Laplace Smoothing (Add-1)

To handle **zero-probability events** (unseen trigrams), we apply Laplace smoothing:

$$P_{\text{smooth}}(w_3 \mid w_1, w_2) = \frac{C(w_1, w_2, w_3) + 1}{C(w_1, w_2) + V}$$

Where:
- $V$ = vocabulary size (total unique tokens)
- The $+1$ in the numerator prevents zero probabilities
- The $+V$ in the denominator normalizes the distribution

---

### Sentence Probability

For a sentence $S = w_1, w_2, \ldots, w_n$, the joint probability under the trigram model is:

$$P(S) = \prod_{i=3}^{n} P(w_i \mid w_{i-2}, w_{i-1})$$

With boundary tokens:

$$P(S) = P(w_1 \mid \langle s \rangle, \langle s \rangle) \cdot P(w_2 \mid \langle s \rangle, w_1) \cdot \prod_{i=3}^{n} P(w_i \mid w_{i-2}, w_{i-1}) \cdot P(\langle /s \rangle \mid w_{n-1}, w_n)$$

---

### Perplexity

**Perplexity** measures how "surprised" the model is by a sentence. Lower values indicate higher confidence:

$$\text{PP}(S) = P(S)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{P(S)}}$$

Equivalently, using log probabilities for numerical stability:

$$\text{PP}(S) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log P(w_i \mid w_{i-2}, w_{i-1})\right)$$

Where:
- $N$ = total number of tokens (including boundary tokens)
- A perplexity of $k$ means the model is as uncertain as choosing uniformly among $k$ words

---

### Weighted Random Sampling

Instead of deterministic $\arg\max$ selection, we sample from the conditional distribution:

$$w_{\text{next}} \sim \text{Categorical}\left(\frac{C(w_1, w_2, w)}{\sum_{w'} C(w_1, w_2, w')} \; \forall \; w \in V\right)$$

This introduces **stochasticity** and produces more diverse outputs.

---

## Pipeline

| Stage | Input | Process | Output |
|:------|:------|:--------|:-------|
| **1. Preprocessing** | Raw sentences | Tokenize, lowercase, add `<s>`, `</s>` | Token sequence |
| **2. Model Building** | Token sequence | Extract trigrams, build CFD | $P(w_3 \mid w_1, w_2)$ |
| **3. Generation** | Seed phrase | Iterative weighted sampling | Generated sentence |
| **4. Probability** | Generated sentence | Product of trigram probabilities | $P(S)$ |
| **5. Perplexity** | $P(S)$, length $N$ | $P(S)^{-1/N}$ | PP score |
| **6. Story Synthesis** | Sentence fragments | Gemini LLM | Coherent narrative |

---

## Implementation Details

### Data Structure

```python
# Conditional Frequency Distribution
# Key: (w₁, w₂) context tuple
# Value: FreqDist of following words

trigram_table[("i", "am")]["happy"]  # → count of "i am happy"
trigram_table[("i", "am")].N()       # → total count of "i am ___"
```

### Hyperparameters

| Parameter | Value | Description |
|:----------|:------|:------------|
| `n` | 3 | N-gram order (trigram) |
| `α` | 1 | Laplace smoothing constant |
| `V` | ~49,000 | Vocabulary size (Brown Corpus) |
| `max_len` | 10 | Maximum generation length |

---

## Tech Stack

<div align="center">

| Component | Technology |
|:----------|:-----------|
| NLP Toolkit | NLTK |
| Corpus | Brown Corpus (1M+ words) |
| Story Generation | Google Gemini 2.5 Flash |
| Language | Python 3.12 |

</div>

---

<div align="center">

**Statistical Language Modeling** • Maximum Likelihood Estimation • Laplace Smoothing

</div>
