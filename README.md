# IMDB Review Analysis

A simple **Sentiment Analysis** project that classifies IMDB movie reviews as **Positive** or **Negative** using **Logistic Regression** and a clean **Tkinter GUI**.

This project demonstrates text preprocessing, model training, and deploying a trained model into a graphical application.

---

## 📘 Project Overview

This project is split into two main Python files:

* **`main.py`** → Trains the sentiment analysis model.
* **`predict.py`** → Loads the saved model and vectorizer, then provides a simple GUI for predicting review sentiment.

The model is trained on the [IMDB Dataset of 50,000 Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), available online.

---

## 🧠 Features

* Text preprocessing (lowercasing, punctuation removal, stopword filtering, etc.)
* TF-IDF vectorization
* Logistic Regression classifier
* Tkinter-based graphical interface for easy testing
* Confidence percentage for each prediction

---

## ⚙️ Requirements

Before running the code, install the necessary dependencies:

```bash
pip install pandas scikit-learn nltk joblib
```

If you want to use the GUI:

```bash
pip install tk
```

---

## 📂 Files Structure

```
IMDB-review-analysis/
│
├── main.py               # Trains the model and saves the .pkl files
├── predict.py            # GUI for predicting review sentiment
├── model.pkl             # (Generated) Trained Logistic Regression model
├── vectorizer.pkl        # (Generated) Saved TF-IDF vectorizer
└── README.md             # Project documentation
```

> ⚠️ **Note:** The dataset (`IMDB Dataset.csv`) is **not included** due to its large size. You must download it manually and place it in the project directory.

Download link: [IMDB Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🚀 How to Run

1. **Download** the IMDB dataset CSV file and place it in the same folder as `main.py`.

2. **Train the model**:

   ```bash
   python main.py
   ```

   This will generate two files — `model.pkl` and `vectorizer.pkl`.

3. **Run the GUI application**:

   ```bash
   python predict.py
   ```

4. Enter your review, click **Predict**, and view the sentiment and confidence score.

---

## 🖥️ GUI Preview

The Tkinter interface uses a dark theme with accent colors:

* **Background:** `#1c1c1c`
* **Text:** `#FFF8E7`
* **Positive Output:** `#00FA9A`
* **Negative Output:** `#F08080`

---

## 📊 Model Info

* **Vectorizer:** TF-IDF (max 5000 features)
* **Classifier:** Logistic Regression (`max_iter=1000`)
* **Accuracy:** ~0.88–0.90 (depending on random split)

---

## 👨‍💻 Author

**Mina Youssef Kamal**
GitHub: [MinaYoussefKamal](https://github.com/MinaYoussefKamal)

---

## 🧩 Future Improvements

* Add Streamlit or custom web UI
* Include real-time dataset loading
* Improve preprocessing (lemmatization, stemming, etc.)
* Experiment with deep learning models like LSTMs or BERT
