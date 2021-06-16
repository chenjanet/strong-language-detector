import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import sys

# load dataset in dataframe object
df = pd.read_csv('../data/clean_data.csv', usecols=['text', 'label'])

# get independent variable (text) and dependent variable (label)
text = df['text'].values
label = df['label'].values

# split dataframe into training + testing sets
text_train, text_test, label_train, label_test = train_test_split(text, label, test_size=0.25, random_state=1000)

# initialize vectorizer based on command-line arguments (default)
vectorizer = None

try:
    arg = int(sys.argv[0])
    if arg == 'count':
        vectorizer = CountVectorizer()
    elif arg == 'tf':
        vectorizer = TfidfVectorizer(use_idf=False, norm='l2')
    elif arg == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError('Invalid mode')
except:
    vectorizer = CountVectorizer()

# fit vectorizer to training data and store in vectorizer.pkl file
vectorizer.fit(text_train)
joblib.dump(vectorizer, 'vectorizer.pkl')

# transform training + testing set with vectorizer
X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

# train model using logistic regression model
classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
classifier.fit(X_train, label_train)

# store in model.pkl file
joblib.dump(classifier, 'model.pkl')
