import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    df = pd.read_csv('imdb.csv')
    df.sentiment = df.sentiment.apply(lambda x : 1 if x =='positive' else 0)
    df['kfold'] = -1
    df = df.sample(frac =1).reset_index(drop = True)
    y = df.sentiment.values
    kf = model_selection.StratifiedKFold(n_splits=5)
    for f, (t_, v_) in enumerate(kf.split(X = df, y=y)):
        df.loc[v_, 'kfold'] = f
    for fold_ in range(5):
        train_df = df[df.kfold != fold_].reset_index(drop = True)
        test_df = df[df.kfold == fold_].reset_index(drop = True)
        tfd = TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None)
        tfd.fit(train_df.review)
        xtrain = tfd.transform(train_df.review)
        xtest = tfd.transform(test_df.review)
        model = linear_model.LogisticRegression(solver ='liblinear')
        model.fit(xtrain, train_df.sentiment)
        preds = model.predict(xtest)
        accuracy = metrics.accuracy_score(test_df.sentiment, preds)
        print(f'fold : {fold_}')
        print(f'accuracy : {accuracy}')
        print('')
        