import sys
import pandas as pd 
import numpy as np

from sqlalchemy import create_engine

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier

from sklearn.metrics import classification_report

nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords


def load_data(database_filepath):
    '''
    A function to load our cleansed data from the database
    INPUT: 
        database_filename: The table name of the prepared data
    OUTPUT: 
         X : Messages data only
         Y : Targets (binary output for each category)
         category_names : Names of each feature in Y
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name='FinalTable',
                           con=engine)
    
    X = df.iloc[:,df.columns.isin(['message'])].message.copy()
    Y = df.iloc[:,~df.columns.isin(['id','message','original','genre'])].copy()
    category_names = list(Y.columns)
    return X, Y, category_names

def tokenize(text):
    '''
    This function is to tokenize the messages in our data, it acts as a 
    feature extractor to be used in training our model.
    
    INPUT: 
        text: Raw text (message)
    OUTPUT: 
         clean_tokens : Text tokenized and prepared to be an input to our classifier
         
    '''
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = text.split() # word_tokenize(text)
    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    '''
    This function is to build our multi-output classifier (binary outputs for each category)
    INPUT: 
        None
    OUTPUT: 
         best_model : Best pipline model  
    '''
    # choosing the model
    base_model = MultinomialNB()
    # making the base model multiclass predictor 
    classifier_model = MultiOutputClassifier(estimator=base_model)
    
    #building the pipline 
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', classifier_model)
                        ])
    
    # GridSearch to find the best parameters
    
    # parameters
    alphas = np.array([1,0.1,0.01,0.001,0])
    
    # Search for best parameters
    grid = GridSearchCV(estimator=pipeline, 
                        param_grid=dict(clf__estimator__alpha=alphas))
    
    # grid.predict() --> Call predict on the estimator with the best found parameters
    
    return grid


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    This function is to test our trained model, it will calculate 
    the accuracy across all categories outputs
    
    INPUT: 
        model : The trained model
        X_test : Test features
        Y_test : Test Targets 
        category_names : Names of our targets
    OUTPUT: 
         An accuarcy results
         
    '''
    # model predictions 
    Y_pred = model.predict(X_test)
    # changing the structure output 
    Y_pred = pd.DataFrame(data=Y_pred,
                          columns=Y_test.columns)
    # total accuarcy
    print('Total accuarcy (all targets): ', np.mean((Y_pred.values) == (Y_test.values))) 


def save_model(model, model_filepath):
    '''
    This function is to save the trained model to be deployed in production later 
    
    INPUT: 
        model : The trained model
        model_filepath : The path where you want to store the trained model
        
    OUTPUT: 
         None
         
    '''
    import pickle
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