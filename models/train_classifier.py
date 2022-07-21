#Importing Libraries
import sys
import nltk
import re
import numpy as np
import pandas as pd
import sqlite3
import sqlalchemy
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
import pickle
nltk.download(['punkt', 'wordnet'])

def load_data(database_filepath):
    """
    Loads data from SQL Lite database.
    
    Parameters:
    database_filepath: Filepath to the database
    
    Returns:
    X: Features (independent variables)
    Y: Target (dependent variable)
    """
    # load data from database 
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("disaster_resp_msgs", con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    return X,Y


def tokenize(text):
    """
    This function Tokenizes and Lemmatizes the given text.
    
    Parameters:
    text: Text to be tokenized
    
    Returns:
    cleaned_tokens: Returns tokens that are lemmatized , Case Normalized (to lowercase) and stripped of leading/trailing white spaces  
    """
    # Tokenize the input text
    tokens = word_tokenize(text)
    
    # Instantiate Lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    #Iterate through each token and lemmatize, normalise case, and remove leading/trailing white space
    cleaned_tokens=[]
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        cleaned_tokens.append(clean_tok)
        
    return cleaned_tokens


def build_model():
    """
    Builds MultiOuptputClassifier and tunes model using GridSearchCV.
    
    Returns:
    cv: Classifier 
    """    
    #initialize machine learning pipeline 
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)), # override default tokenizer with our custome tokenize function
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
        'clf__estimator__n_estimators' : [50, 100]
    }
    
    # create a grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates model's performance and returns classification report of each column. 
    
    Parameters:
    model: classifier
    X_test: test dataset
    Y_test: labels for test data in X_test
    
    Returns:
    Classification report for each column
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(Y_test):
        print(column, classification_report(Y_test[column], y_pred[:, index]))


def save_model(model, model_filepath):
    """ 
    Exports the final model as a pickle file.
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

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