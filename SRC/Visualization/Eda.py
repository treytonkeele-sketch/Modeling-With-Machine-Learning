import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





def perform_eda(train):
    """Perform exploratory data analysis."""
    print("Shape:", train.shape)
    print(train.dtypes)
    print(train.describe())
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length', 'Last Payment Date', 'Customer Status', 'Last Due Date']
    for col in categorical_cols:
        print(f"\nValue counts for {col}:")
        print(train[col].value_counts())