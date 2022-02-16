# import libraries
import sqlite3
import pickle

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import sys 
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report

def load_data(database_filepath):
    # connect to the database
    conn = sqlite3.connect(database_filepath )

    # load data from database
    df = pd.read_sql('SELECT * FROM DisasterMsg', conn)
    
    category_names = df.columns[2:]
    X = df['message']
    y = df[category_names]
    
    return X, y, category_names


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    '''
    Transforms the text into tokens, lemmatizes it and make some other small transformations to the text before
    returning the clean token

            Parameters:
                    text (str): The full message which needs to be treated

            Returns:
                    clean_tokens (list): A list of clean tokens (str)
    '''
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    Creates the model pipeline in order to facilitate standard treatment of the feature and model
    optimization. The basic pipeline consits of a CountVectorizer, with a TfidTransformer and a
    MultiOutputClassifier which runs on a RandomForestClassifier.

            Returns:
                    cv (GridSearchCV): returns an optimized model
    '''    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    # specify parameters for grid search
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        'tfidf__use_idf': (True, False),
        #'clf__n_estimators': [50, 100, 200],
        #'clf__min_samples_split': [2, 3, 4],
        #'features__transformer_weights': (
        #    {'text_pipeline': 1, 'starting_verb': 0.5},
        #    {'text_pipeline': 0.5, 'starting_verb': 1},
        #    {'text_pipeline': 0.8, 'starting_verb': 1},
        #)
    }

    # create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Given a model, the independent variables and the dependent variables and the category names it displays the model results for each disaster category in the dataset

            Parameters:
                    model (GridSearchCV): the optimized model
                    X_test (str) : The text which we are are trying to predict to which categories it belongs
                    Y_test (DataFrame) : A DataFrame with the categories which we will be trying to predict
                    category_names (list) : A list of the category names so we can print the classification report for each category
    '''
    prediction = model.predict(X_test)

    for index, category in enumerate(category_names):
        print('Classification report for category "{}":'.format(category))
        print('{}'.format(classification_report(Y_test[category], prediction[:,index])))


def save_model(model, model_filepath):
    '''
    Saves the model in a pickle format to a file_path

            Parameters:
                    model (GridSearchCV) : The trained model which will be saved to the disk to be used later
                    model_filepath (str) : The file path where the pickle file will be stored
    '''
    # save the model to disk
    pickle.dump(model, open(model_filepath, 'wb'))


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