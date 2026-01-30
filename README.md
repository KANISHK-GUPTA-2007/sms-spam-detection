# 📩 SMS Spam Detection (Binary Text Classification)

A TensorFlow-based machine learning project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using natural language processing and deep learning.

This project demonstrates the complete workflow of a text classification system including data preprocessing, tokenization, model training, evaluation, and prediction.

---

## 🚀 Features

* Binary classification: **Spam vs Ham**
* Built using **TensorFlow 2.15 / Keras**
* Efficient text preprocessing and vectorization
* Supports model saving & reuse
* Easy to extend for other text datasets

---

## 🧠 Model Overview

The project uses:

* Text tokenization & padding
* Embedding layer for word representation
* Neural network classifier (Dense / LSTM / or CNN depending on implementation)
* Binary cross‑entropy loss
* Accuracy as primary evaluation metric

---

## 🗂️ Project Structure (Typical)

```
project/
│
├── data/                  # Dataset (spam.csv or similar)
├── model/                 # Saved models
├── requirements.txt       # Dependencies
└── README.md
```

*(Structure may vary based on your implementation.)*


## ⚙️ Installation

### 1️⃣ Create virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

---

## 🏃 How to Run

### Train the model

```bash
python train.py
```

### Predict new SMS

```bash
python predict.py
```

Example input:

```
Congratulations! You won a free prize
```

Output:

```
Spam
```

---

## 📊 Dataset

Common dataset used:

* SMS Spam Collection Dataset (UCI)
* Contains ~5,500 labeled messages

Format:

```
label,text
ham,Hello how are you
spam,Win cash now!!!
```

---

## 🛠️ Technologies Used

* Python 3.10+
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit‑learn
* Matplotlib (optional for visualization)

---

## 🎯 Future Improvements

* Add attention mechanism
* Use pretrained embeddings (GloVe / FastText)
* Deploy using Flask / FastAPI
* Convert to REST API
* Add confusion matrix & ROC curve

---

## 📜 License

This project is open‑source and free to use for educational purposes.

---

## 🙌 Author

**Kanishk**
Student | Machine Learning Enthusiast
If you find this project useful, consider giving it a ⭐ on GitHub!
