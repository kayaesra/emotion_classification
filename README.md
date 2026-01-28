#  Emotion Classification: A Comprehensive Benchmark Study

This repository presents a rigorous comparative analysis of **Emotion Classification** on textual data. The project explores the spectrum of Natural Language Processing, from traditional statistical methods (**TF-IDF**) to Deep Learning (**BERT**) and modern generative paradigms (**LLM Prompting**).

---

##  1. Project Objectives
This project explores emotion classification from text using multiple approaches, ranging from classical machine learning models to large language models (LLMs) and fine-tuned transformer-based architectures.
The main objective is to compare traditional NLP pipelines with modern transformer-based methods and analyze their strengths and limitations on the same dataset.
---

##  2. Dataset Insight: `dair-ai/emotion`
We utilized the **Emotion** dataset from Hugging Face, containing 20,000 labeled Twitter samples.
* **Emotional Classes:** Sadness (0), Joy (1), Love (2), Anger (3), Fear (4), Surprise (5).
* **Data Split:** * Training: 16,000
  * Validation: 2,000
  * Testing: 2,000
* **Analysis of Imbalance:** The dataset is heavily skewed towards *Joy* and *Sadness*. This project specifically analyzes how this skewness affects the F1-Scores of minority labels like *Surprise* and *Love*.

---

## 3. Technical Methodology & Implementation

### A. Statistical Baseline (Classical ML)
To establish a high-performance floor, we implemented:
* **Feature Extraction:** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization.
* **Logistic Regression:** Achieved an excellent **86% Accuracy**.
* **Support Vector Machine (SVC):** Reached **85% Accuracy**, effectively handling non-linear boundaries in the feature space.

### B. Generative AI Pipeline (LLM Prompting)
We leveraged the **Qwen2.5-1.5B** open-source model using two distinct strategies:
* **Zero-Shot Prompting:** The model was asked to classify text based only on its pre-trained understanding.
* **Few-Shot Prompting:** Context was enriched with specific class exemplars to steer model behavior.
* **Optimization:** Used **Batch Processing (BS: 32)** and **NVIDIA A100 GPU** acceleration to maximize throughput.

### C. Deep Learning (Transformer Based)
* **BERT Embeddings:** Feature extraction using `bert-base-uncased` to capture semantic nuances.
* **Task-Specific Fine-Tuning:** The fine-tuning approach proved to be the most reliable for specific emotional alignment.

---

## 4. Comprehensive Performance Metrics

| Approach             | Performance              |
| -------------------- | ------------------------ |
| TF-IDF + LR          | Strong baseline          |
| TF-IDF + SVC         | Comparable to LR         |
| Qwen (Zero/Few-Shot) | Underperformed           |
| BERT Embeddings + LR | Moderate                 |
| **Fine-Tuned BERT**  | **Best (~93% Accuracy)** |

---

##  5. Advanced Error Analysis & Findings
* **The "Joy" Bias:** Generative models often mislabel *Love* and *Surprise* as *Joy* due to the high semantic similarity in positive sentiment samples.
* **Prompt Fragility:** LLMs showed high sensitivity to output formatting. Minor changes in the system prompt required complex regular expression-based output cleaning.
* **Hardware Efficiency:** Demonstrates the necessity of high-compute GPUs when running inference on modern LLMs even for small parameter models (1.5B).
* **Supervised Superiority:** The findings conclude that **Task-Specific Fine-Tuning** remains significantly more accurate than general LLM prompting for narrow, fixed-label classification tasks.

---

## 6. Repository Roadmap

* **`emotion_classification_project.ipynb`**: The technical core. Contains data ingestion, cleaning, TF-IDF modeling, and the LLM inference pipeline.
* **`emotion_classification_report.pdf`**: Comprehensive academic report containing methodology, theoretical analysis, and visualizations.
* **`emotion_pred.csv`**: **Master Benchmark Data.** A row-by-row comparison of text samples, Ground Truth, and predictions from all tested models.
* **`train.csv` / `test.csv` / `validation.csv`**: Standardized data partitions ensuring full reproducibility.

---
