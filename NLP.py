import nltk
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import string
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
from matplotlib import pyplot as plt

# Download NLTK resources (uncomment if not already downloaded)
# nltk.download('stopwords')

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()

data = pd.read_csv("SMSSpamCollection.tsv", sep='\t')
data.columns = ['label', 'body_text']

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100

data['body_len'] = data['body_text'].apply(lambda x: len(x) - x.count(" "))
data['punct%'] = data['body_text'].apply(lambda x: count_punct(x))

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text

X_train, X_test, y_train, y_test = train_test_split(data[['body_text', 'body_len', 'punct%']], data['label'], test_size=0.2)

tfidf_vect = TfidfVectorizer(analyzer=clean_text)
tfidf_vect_fit = tfidf_vect.fit(X_train['body_text'])
tfidf_train = tfidf_vect_fit.transform(X_train['body_text'])
tfidf_test = tfidf_vect_fit.transform(X_test['body_text'])

X_train_vect = pd.concat([X_train[['body_len', 'punct%']].reset_index(drop=True), pd.DataFrame(tfidf_train.toarray())], axis=1)
X_test_vect = pd.concat([X_test[['body_len', 'punct%']].reset_index(drop=True), pd.DataFrame(tfidf_test.toarray())], axis=1)

def compute(x_input, y_input, x_test):
    index = []
    accuracy = []
    error = []
    for K in range(30):
        K = K + 1
        neigh = KNeighborsClassifier(n_neighbors=K)
        neigh.fit(x_input, y_input)
        y_pred = neigh.predict(x_test)
        index.append(K)
        accuracy.append(accuracy_score(y_test, y_pred) * 100)
        error.append(mean_squared_error(y_test, y_pred) * 100)

    plt.subplot(2, 1, 1)
    plt.plot(index, accuracy)
    plt.title('Accuracy')
    plt.xlabel('Value of K')
    plt.ylabel('Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(index, error, 'r')
    plt.title('Error')
    plt.xlabel('Value of K')
    plt.ylabel('Error')
    plt.tight_layout() # Adjust subplot parameters to give specified padding
    plt.show()

compute(X_train_vect, y_train, X_test_vect)