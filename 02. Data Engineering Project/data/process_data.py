import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads both the messages and the categories files and merge them together droping unused features

            Parameters:
                    messages_filepath (str): String representing the messages file path
                    categories_filepath (str): String representing the categories file path

            Returns:
                    df (DataFrame): Returns the target DataFrame
    '''
    # load messages dataset
    messages_df = pd.read_csv(messages_filepath, index_col=0)
    
    # load categories dataset
    categories_df = pd.read_csv(categories_filepath, index_col=0)
    
    # merge datasets
    df = pd.merge(messages_df, categories_df, left_index=True, right_index=True)
    df = df.drop(columns='original')
    
    return df


def clean_data(df):
    '''
    Cleans up the dataframe by separating each category by column and removing duplicates

            Parameters:
                    df (DataFrame): DataFrame with all categories in one column separated by ';'

            Returns:
                    df (DataFrame): Returns the "cleaned up" DataFrame
    '''
    # create a dataframe of the individual category columns
    categories_df = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories_df.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing. I preferred using split instead.
    category_colnames = row.apply(lambda x: x.split('-')[0]).values
    
    # rename the columns of `categories`
    categories_df.columns = category_colnames
    
    for column in categories_df:
        # set each value to be the last character of the string
        categories_df[column] = categories_df[column].str[-1]
        
        # convert column from string to numeric
        categories_df[column] = pd.to_numeric(categories_df[column])
        
    # drop the original categories column from `df`
    df = df.drop(columns='categories')
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.merge(df, categories_df, left_index=True, right_index=True)
    
    # check number of duplicates (removing the first)
    duplicated_rows = df.index.duplicated()
    print('    Number of rows: {}'.format(df.shape[0]))
    print('    Duplicate rows: {}'.format(duplicated_rows.sum()))
    while duplicated_rows.sum() > 0:
          print('    Removing duplicate rows.')
          df = df[~duplicated_rows]
          duplicated_rows = df.index.duplicated()
          #print('\tDuplicate rows: {}'.format(duplicated_rows.sum()))
    print('    Number of rows: {}'.format(df.shape[0]))
    
    # Replaces other values in related by 1
    df['related'] = df['related'].apply(lambda x: x if ((x==0) or (x==1)) else 1)
    
    return df


def save_data(df, database_filename):
    '''
    Saves the DataFrame into a sqlite database

            Parameters:
                    df (DataFrame): Cleaned up dataframe
                    database_filename (str): String representing the database filename (which will add the extension '.db')
    '''
    engine = create_engine('sqlite:///' + database_filename + '.db')
    df.to_sql('DisasterMsg', engine, index=False, if_exists='replace')

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