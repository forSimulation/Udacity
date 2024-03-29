import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load csv datasets from messages_filepath and categories_filepath.
    Merge the two sets to a dataframe.
    Input: two CSV files' path
    Output: A dataframe contain all infamation.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on=['id'])
    return df

def clean_data(df):
    """
    transform the input raw df to a clean df with no duplicataion
    and label categories with 0 or 1.
    Input : uncleaned dataframe
    Output: cleaned dataframe
    """
    # Split categories into separate category columns.
    categories = df.categories.str.split(pat=';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = list(map(lambda x:x[0:-2],row))
    # rename the columns of `categories`
    categories.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        # categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        # categories[column] = pd.to_numeric(categories[column])
        categories[column] = categories[column].astype(int)
    # Replace categories column in df with new category columns
    # drop the original categories column from `df`
    df.drop(labels ='categories',axis = 'columns', inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(categories)
    # Remove duplicates.
    # drop duplicates
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    return df



def save_data(df, database_filename):
    """
    save the dateframe to a sqlite database.
    Input: cleaned dataframe.
    Output: a sqlite database
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disasterResponse', engine, index=False)  


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