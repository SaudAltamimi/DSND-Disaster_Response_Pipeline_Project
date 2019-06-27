import sys
import pandas as pd 
import sqlite3
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages
    messages = pd.read_csv(messages_filepath)
    #load categories
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(left=messages,right=categories,on='id',how='left')
    
    return df
def clean_data(df):
    # take categories & IDs
    categories = df[['id','categories']]
    #Split categories into separate category columns.
    categories_splitted = categories.categories.str.split(pat=';',
                                                      expand=True)
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(data=categories_splitted.values,
                              index=categories.id)
    categories.reset_index(inplace=True)
    # extract a list of new column names for categories.
    category_colnames = [i.split('-')[0] for i in categories_splitted.head(1).values[0,0:]]
    # rename the columns of `categories`
    categories.columns = ['id'] + category_colnames
    # Convert category values to just numbers 0 or 1
    for column in categories.columns[1:]:#without the id column
        # set each value to be the last character of the string
        categories[column] = categories[column].str.slice(start=-1,step=1)
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Replace categories column in df with new category columns.
    # drop the original categories column from `df
    df.drop(axis='columns',columns='categories',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(left=df,right=categories)

    # removing duplicates 
    df.drop_duplicates(keep='first',inplace=True)
    
    return df
def save_data(df, database_filename):
    # Save the clean dataset into an sqlite database.
    engine = create_engine('sqlite:///' + database_filename)
    
    df.to_sql('FinalTable', engine,
              index=False,
              if_exists='replace')
    
    # print a sample
    print('\n Sample: \n')
    print(df.sample().to_string())


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()