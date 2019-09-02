import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import re
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.e
import nltk
# download files of NLTK
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
    load sqlite datasets and seperate the features and targets.
    Input: filepath name
    Output: 
    X: fetures used to be learned contain message.
    Y: targets to be classfied. contain all categories of messages.
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('disasterResponse', engine)
    X = df['message']
    Y = df.loc[:, 'related':'direct_report']
    return X, Y

def tokenize(text):
    """
    transform plain text into important words sequence.
    Input: raw text
    Output: lemmed text
    """
    # Normalize text
    text_nor = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text_nor)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their stems
    stemmed = [PorterStemmer().stem(w) for w in words]
    # Reduce words to their root form
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    # Lemmatize verbs by specifying pos
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed

def build_model():
    """
    Using pipline to build AdaBoost model to classfy messages.
    grid search is used to determine parameters.
    Input: none
    Output: optimal model
    """
    pipeline =  Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(random_state=6)))
    ])
    # grid search 
    parameters = {
    'vect__ngram_range': ((1, 2),(1, 3)),
    'clf__estimator__learning_rate': (0.5, 0.1),
    'clf__estimator__n_estimators': (160, 220)
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=8)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the targets' performance on test data in category_names 
    Input: model: model to be evaluated, X_test: test features, Y_test: test targets, 
           category_names: print targets.
    Output: print accuracy, precision, recall and f1 score on default show device.
    """
    Y_to_test = model.predict(X_test)
    Y_to_test_df = pd.DataFrame(Y_to_test)
    Y_to_test_df.columns = Y_test.columns
    for name in category_names:
        accuracy = accuracy_score(Y_test.loc[:, name], Y_to_test_df.loc[:, name])
        precision = precision_score(Y_test.loc[:, name], Y_to_test_df.loc[:, name], average='micro')
        recall = recall_score(Y_test.loc[:, name], Y_to_test_df.loc[:, name], average='micro')
        f1 = f1_score(Y_test.loc[:, name], Y_to_test_df.loc[:, name], average='micro')
        print("%-22s |accuracy: %.3f |  precision: %.3f | recall: %.3f | f1: %.3f |\n" % (name, accuracy, precision, recall, f1))

def save_model(model, model_filepath):
    """
    Export your model as a pickle file 
    Input: model: model to be saved, model_filepath: saved file path 
    Output: pickle file of model.
    """
    file = open(model_filepath, 'wb')
    pickle.dump(cv, file)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()