import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("train.csv")

CONTRACTIONS_DICT = {
    "can`t": "can not",
    "won`t": "will not",
    "don`t": "do not",
    "aren`t": "are not",
    "i`d": "i would",
    "couldn`t": "could not",
    "shouldn`t": "should not",
    "wouldn`t": "would not",
    "isn`t": "is not",
    "it`s": "it is",
    "didn`t": "did not",
    "weren`t": "were not",
    "mustn`t": "must not",
}

def replace_words(string:str, dictionary:dict):
    for k, v in dictionary.items():
        string = string.replace(k, v)
    return string

def prepare_data(df:pd.DataFrame):
    df["text"] = df["text"].apply(lambda x: re.split('http:\/\/.*', str(x))[0]).str.lower().apply(lambda x: replace_words(x, CONTRACTIONS_DICT))
    df["label"] = df["sentiment"].map({"neutral": 1, "negative":0, "positive":2})
    return df.text.values, df.label.values

train_tweets, train_labels = prepare_data(train_df)
test_tweets, test_labels = prepare_data(test_df)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_tweets)
train_tokenized = tokenizer.texts_to_matrix(train_tweets, mode='tfidf')
test_tokenized = tokenizer.texts_to_matrix(test_tweets, mode='tfidf')

forest = RandomForestClassifier(n_estimators=500, min_samples_leaf=2, oob_score=True, n_jobs=-1)
forest.fit(train_tokenized, train_labels)

print(f"Train score: {forest.score(train_tokenized, train_labels)}")
print(f"OOB score: {forest.oob_score_}")
print(f"Test score: {forest.score(test_tokenized, test_labels)}")

plot_confusion_matrix(forest, test_tokenized, test_labels, display_labels=["Negative","Neutral","Positive"], normalize='true')
