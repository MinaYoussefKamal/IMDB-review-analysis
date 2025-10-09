#------------Training--------------#


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from nltk.corpus import stopwords
import joblib

data=pd.read_csv("IMDB Dataset.csv")
stop_words = set(stopwords.words('english'))
data['review'] = data['review'].str.lower()
data['review'] = data['review'].replace(r'[^\w\s]', '', regex=True)
data['review'] = data['review'].replace(r'(<br\s*/?>|br\s*/?|/br)', '', regex=True)
data['review'] = data['review'].apply(lambda x: " ".join([word for word in x.split() if word not in stop_words]))

x= data["review"]
y = data["sentiment"].apply(lambda x:1 if x== "positive" else 0)

x_train, x_test, y_train, y_test = train_test_split(x, y ,test_size=0.2, random_state=42)


vectorizer= TfidfVectorizer(max_features=5000)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)

print("Accuracy: ", accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred,target_names=["negative", "positive"]))

joblib.dump(model, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("✅ Model and vectorizer saved successfully!")
