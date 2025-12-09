import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

def optimal_k(data):   
    n = data.shape[0]
    k = int(np.sqrt(n))
    if k % 2 == 0:
        k += 1
    return k    

def train_knn_model(train, test):
    """Train KNN model and make predictions."""
    obj_cols = train.select_dtypes(include=['object']).columns
    train = pd.get_dummies(train, columns=obj_cols, drop_first=True)
    test = pd.get_dummies(test, columns=obj_cols, drop_first=True)

    X = train[['Support Calls', 'Contract Length_Monthly', 'Age']].copy()
    y = train['Churn'].copy()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=67)

    k = optimal_k(X_train)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsRegressor(n_neighbors=k))
    ])
    pipeline.fit(X_train, y_train)

    X_test = test[['Support Calls', 'Contract Length_Monthly', 'Age']].copy()
    test_pred = pipeline.predict(X_test)
    submission = pd.DataFrame({
        'CustomerID': test['CustomerID'].values,
        'Churn': test_pred
    })
    submission.to_csv('Submission.csv', index=False)
    print("Wrote Submission.csv. Preview:")
    print(submission.head())
    return pipeline, X_val, y_val, submission