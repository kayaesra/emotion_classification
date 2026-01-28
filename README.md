# 🎭 Emotion Classification: Classical ML vs. Large Language Models

This repository contains a comprehensive benchmark study on **Emotion Classification**, comparing traditional feature engineering methods with modern Large Language Model (LLM) prompting techniques.

## 🎓 Academic Context
* **Institution:** Yeditepe University
* **Course:** MTH424 - Generative AI Models
* **Project Term:** Spring 2025
* **Lead Developer:** **Esra Kaya**

---

## 📌 Project Overview
The primary goal of this research is to evaluate whether "prompt-only" inference using an open-source LLM (**Qwen2.5-1.5B**) can compete with task-specific supervised learning models. The study involves a multi-class classification of six distinct emotional states.

### 📊 Dataset: `dair-ai/emotion`
* **Size:** 20,000 samples (16,000 Training / 2,000 Validation / 2,000 Test).
* **Labels:** Sadness (0), Joy (1), Love (2), Anger (3), Fear (4), Surprise (5).
* **Key Challenge:** The dataset is **unbalanced**, with Joy and Sadness being dominant. This distribution highlights the difficulty LLMs face with minority classes like "Surprise".

---

## 🛠️ Technical Methodology

### 1. Classical Machine Learning (Supervised Baselines)
Implemented using **TF-IDF Vectorization** to establish high-accuracy benchmarks:
* **Logistic Regression:** Achieved **86% test accuracy**.
* **Support Vector Classifier (SVC):** Achieved **85% test accuracy**.

### 2. Large Language Model (Generative Inference)
Explored the classification capabilities of **Qwen2.5-1.5B**:
* **Zero-Shot Prompting:** Testing the model's direct reasoning without prior examples.
* **Few-Shot Prompting:** Guiding the model with specific class exemplars.
* **Optimization:** Leveraged **NVIDIA A100 GPU** and **Batch Processing (Size: 32)** for efficient, large-scale inference.

### 3. Advanced Transformer Techniques
* **Embedding-Based Models:** Feature extraction using `bert-base-uncased`.
* **Fine-Tuning:** Applied task-oriented fine-tuning to reach maximum semantic alignment.

---

## 📈 Results & Key Findings

| Methodology | Model | Test Accuracy | Macro F1-Score |
| :--- | :--- | :---: | :---: |
| Supervised | **BERT Fine-Tuning** | **Highest** | **High** |
| Supervised | **Logistic Regression** | 0.86 | 0.79 |
| Supervised | **SVC (TF-IDF)** | 0.85 | 0.78 |
| Prompting | **Qwen2.5 (Zero-Shot)** | 0.45 | 0.38 |
| Prompting | **Qwen2.5 (Few-Shot)** | 0.24 | 0.17 |

### 🔍 Analytical Insights
* **Task Specialization:** The results confirm that for narrow classification tasks, **fine-tuned small models** significantly outperform general-purpose prompting on 1.5B parameter models.
* **Prompt Sensitivity:** LLMs required extensive output cleaning and specific prompt engineering to adhere to numerical label formats.
* **Data Imbalance:** Minority classes (Love, Surprise) showed higher error rates, emphasizing the need for balanced training data.

---

## 📂 Project Structure

* **`emotion_classification_project.ipynb`**: Complete development pipeline including data preprocessing, training, and LLM inference.
* **`emotion_classification_report.pdf`**: Detailed academic report covering methodology, error analysis, and conclusions.
* **`emotion_pred.csv`**: **Master Inference File.** A row-by-row comparison of Ground Truth labels against all model predictions (LR, SVC, BERT, Fine-Tuning).
* **`train.csv` / `test.csv` / `validation.csv`**: The standardized dataset splits used for training and benchmarking.

---
