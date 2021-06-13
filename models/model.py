import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# load dataset in dataframe object
df = pd.read_csv('../data/clean_data.csv', usecols=['text', 'label'])

# get independent variable (text) and dependent variable (label)
text = df['text'].values
label = df['label'].values

# split dataframe into training + testing sets
text_train, text_test, label_train, label_test = train_test_split(text, label, test_size=0.25, random_state=1000)
vectorizer = CountVectorizer()
vectorizer.fit(text_train)

# store vectorizer in vectorizer.pkl file
joblib.dump(vectorizer, 'vectorizer.pkl')

# use bag of words model for text
X_train = vectorizer.transform(text_train)
X_test = vectorizer.transform(text_test)

# train model using logistic regression model
classifier = LogisticRegression(solver='lbfgs', max_iter=1000)
classifier.fit(X_train, label_train)

# store in model.pkl file
joblib.dump(classifier, 'model.pkl')
