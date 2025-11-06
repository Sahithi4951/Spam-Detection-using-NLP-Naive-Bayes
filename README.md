## 📧 Spam Detection using NLP & Naive Bayes

This project uses Natural Language Processing (NLP) and Machine Learning to classify SMS messages as Spam or Ham (Not Spam) using the Naive Bayes Algorithm.
It’s a classic example of text classification using the Bag of Words (TF-IDF) model.

---

## 🚀 Features

Cleans and preprocesses text data (removes punctuation, numbers, stopwords, and applies stemming)
Converts text into numerical features using TF-IDF Vectorization
Trains a Multinomial Naive Bayes classifier
Evaluates model performance with Accuracy, Confusion Matrix, and Classification Report
Allows user input to test real-time message classification

---

## 📂 Dataset

The dataset used:
spam.csv

Columns:

v1 → Label (ham or spam)
v2 → Message text

---

## 🧠 Model Workflow

Load and clean the dataset
Encode labels (Spam → 1, Ham → 0)
Preprocess text (lowercasing, punctuation removal, stopword removal, stemming)
Convert text into numerical vectors using TF-IDF
Split data into training and testing sets
Train Multinomial Naive Bayes model
Evaluate performance
Predict for custom user input

---

## 📊 Model Performance

Accuracy: ~97%
Confusion Matrix:

[[965   0]
 [ 30 120]]


The model performs well at identifying spam messages.

---

## 💬 Example Prediction
Enter a message to check if it's spam or not: 
"Congratulations! You’ve won a $1000 Walmart gift card. Click here to claim!"

🚨 This message is SPAM!
Confidence -> HAM: 3.21%, SPAM: 96.79%

Enter a message to check if it's spam or not: 
"Hey, are we meeting for lunch today?"

💌 This message is HAM (not spam).
Confidence -> HAM: 98.34%, SPAM: 1.66%

---

## 🧰 Libraries Used

pandas
numpy
nltk
scikit-learn
re (regex for cleaning text)

---

## 🧼 Text Preprocessing Steps

Lowercasing
Removing punctuation and numbers
Removing stopwords using NLTK
Applying stemming using Porter Stemmer

---

## ✨ Author

Sahithi Bashetty
📧 bashettysahithi@gmail.com