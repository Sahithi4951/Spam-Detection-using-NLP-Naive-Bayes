import pandas as pd
import numpy as np
import string
import re
from nltk.stem import PorterStemmer
from sklearn.preprocessing import LabelEncoder
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


data = pd.read_csv("C:/Users/bashe/Downloads/archive (4)/spam.csv", encoding='latin1')

# Renaming the columnsṇ
data = data.rename(columns = { "v1" : "label" , "v2" : "message"})

# Removing unnecessary columns
data = data.drop(columns=["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"])

# Encoding the text into numeric
lab = LabelEncoder()
data["label"] = lab.fit_transform(data["label"])

# cleaning the data
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation/numbers
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word not in stop_words]

    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]  # stemming once
    return ' '.join(words)
cleaned_text = []
for i,row in data.iterrows():
    text = row['message']
    clean = clean_text(text)
    cleaned_text.append(clean)
data['cleaned_text'] = cleaned_text

# vectorizing the texts
vectorizer = TfidfVectorizer(max_features = 3000)
X = data['cleaned_text']
y = data['label']

# training and test the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Creating the user input
y_pred = model.predict(X_test_tfidf)
message = input("Enter your message here:")
cleaned = clean_text(message)
X_input = vectorizer.transform([cleaned])
pred = model.predict(X_input)[0]
if pred==1:
     print("🚨 This message is SPAM!")
else:
    print("💌 This message is HAM (not spam).")

# printing the accuaray, Confusion matrix and Classification report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))