# 🚀 TrendSense AI

### AI-Powered Social Media Sentiment, Emotion, Topic & Trend Analysis

TrendSense AI is an NLP-powered analytics dashboard that analyzes social media content to understand **sentiment, emotions, trending topics, engagement patterns, and overall audience behavior**.

Built as an AI & Data Science project using Python, NLP, Machine Learning, and Streamlit.

---

## ✨ Features

* 📊 **Sentiment Analysis** — Classifies posts as Positive, Neutral, or Negative
* 💭 **Emotion Detection** — Detects emotions such as Joy, Sadness, Anger, Fear, Love, and Surprise
* 🔥 **Topic Modeling** — Identifies major discussion themes using LDA
* 📈 **Trend & Engagement Analysis** — Analyzes likes, shares, and engagement
* 🔎 **Post Explorer** — Search and inspect individual social media posts
* 📄 **Automated Reports** — Export analysis as CSV, Excel, and PDF
* 🧪 **Model Performance** — Accuracy, Precision, Recall, F1/F2 scores, confusion matrices, and classification reports
* 🎨 **Interactive Dashboard** — Built with Streamlit and Plotly

---

## 🧠 NLP & Machine Learning

### Sentiment Analysis

* Dataset: TweetEval Sentiment
* TF-IDF Vectorization
* Logistic Regression
* N-grams: Unigrams + Bigrams
* Classes: Positive, Neutral, Negative

### Emotion Detection

* Dataset: dair-ai Emotion
* TF-IDF Vectorization
* Logistic Regression
* Six emotion classes

### Topic Modeling

* Latent Dirichlet Allocation (LDA)
* Count Vectorization
* Automatic topic assignment

---

## 🛠️ Tech Stack

**Programming:**
Python

**Data & ML:**
Pandas • NumPy • Scikit-learn • Joblib

**NLP:**
NLTK • TF-IDF • LDA

**Visualization:**
Plotly • Matplotlib • WordCloud

**Dashboard:**
Streamlit

**Deployment:**
Streamlit Community Cloud

---

## 📂 Project Structure

```text
TrendSense-AI/
│
├── data/
│   ├── social_media.csv
│   ├── sentiment_train.csv
│   ├── sentiment_test.csv
│   ├── emotion_train.csv
│   ├── emotion_test.csv
│   └── topic_analyzed_social_media.csv
│
├── app.py
├── nlp_utils.py
├── train_model.py
├── topic_model.py
├── emotion_model.py
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── topic_model.pkl
├── topic_vectorizer.pkl
├── emotion_model.pkl
├── emotion_vectorizer.pkl
│
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Run Locally

Clone the repository:

```bash
git clone https://github.com/shaikhkhushi02-svg/TrendSense-AI.git
cd TrendSense-AI
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Use Cases

TrendSense AI can be used for:

* 📱 Social media monitoring
* 📢 Brand sentiment tracking
* 📈 Trend discovery
* 🛍️ Customer feedback analysis
* 🎯 Marketing research
* 🧑‍💻 NLP experimentation
* 📊 Audience behavior analysis

---

## 🔮 Future Improvements

* Real-time social media API integration
* Transformer-based models such as BERT/RoBERTa
* Real-time trend detection
* Advanced topic clustering
* Multilingual sentiment analysis
* Brand monitoring
* Automated business insights
* Cloud-based scalable architecture

---

## 👩‍💻 Developer

**Khushi Zehra Shaikh**

AI & Data Science Student | Aspiring Data Engineer

Interested in Artificial Intelligence, Data Engineering, NLP, Machine Learning, and building real-world AI applications.

---

⭐ If you find this project interesting, consider starring the repository!
