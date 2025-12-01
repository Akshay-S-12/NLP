import numpy as np
import pandas as pd
import nltk
import re
from nltk.tokenize import TweetTokenizer, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle

# Download NLTK resources if not already
nltk.download('stopwords')
nltk.download('punkt')

# Load data
df = pd.read_csv("twitter_validation.csv", header=None, encoding='ISO-8859-1')
df.columns = ['id', 'media', 'target', 'text']
df.drop(df.index[(df['target'] == 'Irrelevant')], axis=0, inplace=True)
df.reset_index(drop=True, inplace=True)
df.drop(['id', 'media'], axis=1, inplace=True)
df['target'] = df['target'].map({'Positive': 1, 'Negative': -1, 'Neutral': 0})

tweets = df.text

# Preprocessing
tk = TweetTokenizer()
tweets = tweets.apply(lambda x: tk.tokenize(x)).apply(lambda x: " ".join(x))
tweets = tweets.str.replace('[^a-zA-Z0-9]', ' ', regex=True)
from nltk.tokenize import word_tokenize
tweets = tweets.apply(lambda x: ' '.join([w for w in word_tokenize(x) if len(w) > 2]))
stemmer = SnowballStemmer('english')
tweets = tweets.apply(lambda x: [stemmer.stem(i.lower()) for i in word_tokenize(x)]).apply(lambda x: ' '.join(x))
stop = stopwords.words('english')
tweets = tweets.apply(lambda x: [i for i in word_tokenize(x) if i not in stop]).apply(lambda x: ' '.join(x))

# Vectorization
vec = TfidfVectorizer()
X = vec.fit_transform(tweets)
y = df['target'].values

# This training step would normally use the full dataset, but for inference we use the vectorizer fitted on train
# In a real scenario, save fitted on train data
# But since the notebook used full data, we'll replicate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# Train model
model = SVC()
model.fit(X_train, y_train)

# Save vectorizer and model
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vec, f)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model and vectorizer saved!")
