from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from SRC.Models.Knn_model import train_knn_model
import pandas as pd

train = pd.read_csv('Data/Processed/train_cleaned.csv')

test = pd.read_csv('Data/Processed/test_cleaned.csv')
pipeline, X_val, y_val, submission = train_knn_model(train, test)


def evaluate_model(pipeline, X_val, y_val):
    
    val_pred = pipeline.predict(X_val)
    val_pred_binary = (val_pred >= 0.5).astype(int)
    f1 = f1_score(y_val, val_pred_binary)
    precision = precision_score(y_val, val_pred_binary)
    recall = recall_score(y_val, val_pred_binary)
    roc_auc = roc_auc_score(y_val, val_pred)
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision Score: {precision:.4f}")
    print(f"Recall Score: {recall:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    return f1, precision, recall, roc_auc


# Evaluate the model
results = evaluate_model(pipeline, X_val, y_val)



