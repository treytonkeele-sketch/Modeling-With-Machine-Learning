import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def clean_data(df):
    """Clean the dataset."""
    df['Tenure'] = df['Tenure'].fillna(df['Tenure'].median())
    df['Support Calls'] = df['Support Calls'].replace('none', 0)
    df['Support Calls'] = pd.to_numeric(df['Support Calls'], errors='coerce').fillna(0).astype(int)
    df['Payment Delay'] = df['Payment Delay'].fillna(df['Payment Delay'].mean())
    df['Last Interaction'] = df['Last Interaction'].fillna(df['Last Interaction'].mean())
    return df
