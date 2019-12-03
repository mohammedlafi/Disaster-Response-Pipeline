# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    input
          messages_filepath : the path of the messages.csv file
          categories_filepath   : the path of the categories.csv file
    output
           data frame that is the merge of messages_filepath and  categories_filepath
    '''
     
    # load messages dataset
    messages =  pd.read_csv('disaster_messages.csv')
    # load categories dataset
    categories = pd.read_csv('disaster_categories.csv')
    # merge datasets
    df = messages.merge(categories, how='left', on=['id'])
    

def clean_data(df):
    '''
    input
          data frame that is the merge of messages_filepath and  categories_filepath
    output
        a cleaned data frame in which we removed null values and converted values 
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";", n = 36, expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    split_col = lambda col: col.split('-')
    category_colnames = row.apply(split_col).str.get(0)

    # rename the columns of `categories`
    categories.columns =category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]      
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        # grades[col] = pd.to_numeric(grades[col])
        
    # drop the original categories column from `df`
    df.drop(["categories"], axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    # Place the DataFrames side by side
    df = pd.concat([df, categories], axis=1)
    
    
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    '''
    input
          data frame
    output
         database file
    '''
    engine = create_engine('sqlite:///disaster.db')
    df.to_sql('messages', con=engine, if_exists='replace',index=False)
      


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
