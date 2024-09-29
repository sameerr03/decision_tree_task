# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 12:14:49 2024

@author: samvi
"""

import pandas as pd
import numpy as np

def load_data(path):
    '''
    Loads data from a CSV file

    Parameters
    ----------
    path : str
        file path of the dataset.

    Returns
    -------
    pd.DataFrame
    
    '''
    return pd.read_csv(path)

def drop_missing_observations(df):
    '''
    Drops missing observations(NaN values) from the dataset to ensure no missing value errors are thrown 

    Parameters
    ----------
    df : pd.DataFrame
        Dataset that has been imported as a dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with observations with missing values dropped.

    '''
    return df.dropna()

def preprocess_data(input_file, output_file):
    '''
    Intital processing of the dataset before the models are trained - Handling missing values, creating features from categorical variables

    Parameters
    ----------
    input_file : str
        file path to the original dataset.
    output_file : str
        file path to save the preprocessed dataset.

    Returns
    -------
    None.

    '''
    # Loading the data
    df = load_data(input_file)
    
    # Drop observations with missing values
    df = drop_missing_observations(df)
    
    # Create binary variable for Transfer or Cash Out
    df['is_transfer_or_cashout'] = df['type'].isin(['TRANSFER', 'CASH_OUT']).astype(int)
    
    # Create binary variable if the full amount of the Origin old balance is the amount 
    df['is_full_amount'] = (df['amount'] == df['oldbalanceOrg']).astype(int)

    # Select features for the model
    features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                'is_transfer_or_cashout', 'is_full_amount', 'isFraud']
    
    # Prepare the final dataset
    final_df = df[features]
    
    # Save the preprocessed data
    final_df.to_csv(output_file, index=False)
    
    print(f"Preprocessed data saved to {output_file}")
    print(f"Shape of preprocessed data: {final_df.shape}")

if __name__ == "__main__":
    input_file = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\data\fraud_train.csv"
    output_file = r"C:\Users\samvi\OneDrive\Desktop\Semester\IML\Sameer_Rangwala_A1\decision_tree_task\data\preprocessed_fraud_train.csv"
    preprocess_data(input_file, output_file)