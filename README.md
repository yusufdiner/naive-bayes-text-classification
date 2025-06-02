# BBM409 - Naive Bayes News Classifier

This repository contains my implementation of a Naive Bayes text classifier for predicting the category of English news articles. The project was completed as **Assignment 3** for the course **BBM409 - Introduction to Machine Learning Lab**, Fall 2022.

---

## 🧠 Assignment Overview

### 🗂️ Task
Classify news articles into one of five categories:
- **Sport**
- **Business**
- **Politics**
- **Entertainment**
- **Tech**

### 📌 Goals

- Understand and implement the **Naive Bayes** algorithm from scratch.
- Evaluate different feature representations:
  - Unigram (single words)
  - Bigram (two-word phrases)
- Analyze the effect of stopwords and selected vocabulary (TF-IDF).
- Measure accuracy and interpret the most informative words.

---

## 🧪 Methods

### Part 1: Data Analysis
- Explored keyword frequencies across classes.
- Identified category-specific indicator words.

### Part 2: Naive Bayes Implementation
- Implemented from scratch using **Bag-of-Words** model.
- Applied **Laplace smoothing** and **log probabilities**.
- Handled unseen words in testing data.

### Part 3: TF-IDF & Stopword Analysis
- Selected top 10 words per class that predict presence or absence.
- Compared classification accuracy with and without stopwords.
- Visualized and discussed the impact of these decisions.

### Part 4: Accuracy Calculation
- Used 80-20 train-test split.
- Calculated classification accuracy for each configuration.

---

## 📈 Accuracy Results

| Feature Type | Stopwords Included | Accuracy (%) |
|--------------|--------------------|---------------|
| Unigram      | ✅ Yes              | ...           |
| Unigram      | ❌ No               | ...           |
| Bigram       | ✅ Yes              | ...           |
| Bigram       | ❌ No               | ...           |

---

## 🗃️ Files

- `main.py`: Core implementation including:
  - BoW construction
  - Naive Bayes classification
  - TF-IDF word selection
  - Accuracy evaluation
- `reportcode.ipynb`: Report and analysis with explanations, charts, and outputs.
- `Assignment3_Fall2022_409.pdf`: Assignment description.

---

## ⚙️ Tech Stack

- Python
- NumPy, Pandas
- scikit-learn (only for vectorization & TF-IDF)

No external ML libraries (like sklearn's Naive Bayes) were used for model implementation — it's built entirely from scratch.

---

## 🚀 Run

Make sure `English Dataset.csv` is in the same directory.

```bash
python main.py
