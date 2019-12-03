import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


#----------------------------------
import pickle
import warnings
import string
import unittest
warnings.filterwarnings("ignore")
#----------------------------------


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem         import WordNetLemmatizer

from nltk.corpus import stopwords

nltk.download(['punkt', 'wordnet','stopwords'])


# ------------------------------------------
from nltk.stem.porter import PorterStemmer
#from nltk.stem.wordnet import WordNetLemmatizer
# ------------------------------------------



from sklearn.metrics                 import confusion_matrix
from sklearn.model_selection         import train_test_split
from sklearn.ensemble                import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.multioutput             import MultiOutputClassifier
from sklearn.utils                   import shuffle
from sklearn.datasets                import make_classification
from sklearn.pipeline                import Pipeline, FeatureUnion


# import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection         import train_test_split, GridSearchCV
from sklearn.metrics                 import precision_score, recall_score, f1_score,classification_report, make_scorer
from sklearn.ensemble                import AdaBoostClassifier
from sklearn.base                    import BaseEstimator, TransformerMixin

def load_data(database_filepath):
    '''
    input
          database_filepath : the path of the database
    output
        X : training dataset
        y : test dataset
        
    '''
    ## Execute this code cell to output the values in the categories table
    # connect to the database
    # the database file will be disaster.db

    engine = create_engine('sqlite:///disaster.db')
    # load data from database
    df = pd.read_sql_table('messages', engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(df.columns[4:])

    return X, y , category_names

def tokenize(text):
    '''
    input
          text  : raw text
    output
       tokens : set of words 
        
    '''
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens

def build_pipeline():
    
    pipeline = Pipeline ([
        ('vect'    , CountVectorizer(tokenizer=tokenize)),
        ('tfidf'   , TfidfTransformer()),
        ('clf'     , MultiOutputClassifier(RandomForestClassifier( ) ))     
    ])
    
    return pipeline

def build_model():
    '''
    input
          text  : raw text
    output
       tokens : set of words 
        
    '''
    
    pipeline = Pipeline ([
        ('vect'    , CountVectorizer(tokenizer=tokenize)),
        ('tfidf'   , TfidfTransformer()),
        ('clf'     , MultiOutputClassifier(RandomForestClassifier( ) ))     
    ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    input
    
     output
     
        
    '''
    # predict on the X_test
    y_pred = model.predict(X_test)
    # build classification report on every column
    performances = []
    for i in range(len(category_names)):
        performances.append([f1_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             precision_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro'),
                             recall_score(Y_test.iloc[:, i].values, y_pred[:, i], average='micro')])
    # build dataframe
    performances = pd.DataFrame(performances, columns=['f1 score', 'precision', 'recall'],
                                index = category_names)   
    return performances


def save_model(model, model_filepath):
    '''
    input
    
     output
     
        
    '''
    pickle.dump(model, open(model_filepath, "wb"))



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
