Emotion Classification: Evaluating LLMs vs. Classical ML Approaches
This repository contains the comprehensive study and implementation of Emotion Classification on textual data, comparing traditional machine learning techniques with modern Large Language Models (LLMs) through Zero-Shot and Few-Shot prompting.

** Project Context**
Developed as the Term Project for MTH424 - Generative AI Models at Yeditepe University (Spring 2025).

** Project Objectives**
Evaluate the "prompt-only" performance of open-source LLMs in text classification.

Benchmark LLM performance against established supervised learning baselines (Logistic Regression & SVC).

Analyze the impact of data imbalance and prompt engineering on classification accuracy.

**Dataset: dair-ai/emotion
**We utilized the standardized "emotion" dataset from Hugging Face, which consists of Twitter messages labeled into 6 emotional categories:

Classes: Sadness (0), Joy (1), Love (2), Anger (3), Fear (4), Surprise (5).

Distribution: 16,000 Training / 2,000 Validation / 2,000 Test.

Key Observation: The dataset is unbalanced, with "Joy" and "Sadness" being the most frequent, while "Surprise" and "Love" are significantly underrepresented.

**Technical Implementation & Methodology**
1. Classical Machine Learning (The Baselines)
Using TF-IDF Vectorization, we trained two core models to establish a high-performance baseline:

Logistic Regression: Achieved an impressive 86% accuracy on the test set.

Support Vector Classifier (SVC): Reached 85% accuracy.

Conclusion: These models served as a strong benchmark, showing that specialized supervised learning handles specific text tasks with high precision.

2. LLM Implementation: Qwen2.5-1.5B
We tested the reasoning capabilities of Qwen2.5-1.5B using two distinct prompting strategies:

Zero-Shot Prompting: The model classified text based solely on the label descriptions.

Few-Shot Prompting: We provided the model with context through exemplars for each class to guide its reasoning.

Performance Optimization: We utilized Batch Processing (Batch Size: 32) and A100 GPU acceleration to handle large-scale inference efficiently.

3. Fine-Tuning & Advanced Methods (Extra Credit)
BERT Integration: Implemented bert-base-uncased for feature extraction.

Custom Fine-Tuning: The project demonstrated that a task-specific fine-tuned model (like BERT) often yields superior results compared to general-purpose prompt-based LLMs.

**Key Insights from the Report:**
Prompt Sensitivity: Even minor changes in the prompt significantly altered the LLM's output format, requiring rigorous output cleaning.

Class Confusion: The LLM frequently struggled with the "Surprise" and "Love" classes due to their lower representation in the dataset and semantic similarity to "Joy".

The Power of Fine-Tuning: The results confirmed that training a model for a specific task remains more effective than using a general model with simple prompts for high-accuracy requirements.


**Project Structure**
emotion_classification_project.ipynb: Full implementation including data loading, model training, and LLM inference.
emotion_classification_report.pdf: Detailed academic analysis, methodology, and performance evaluations.
train.csv: The primary training set (16,000 samples) used to build the supervised learning baselines.
test.csv: The final evaluation set (2,000 samples) used to generate the benchmark metrics for both Classical ML and LLM models.
validation.csv: The development set (2,000 samples) used for hyperparameter tuning and prompt refinement.

