from SRC.Data.load_data import load_data
from SRC.Data.Process_data import clean_data
from SRC.Visualization.Eda import perform_eda
from SRC.Visualization.Plots import plot_histograms
from SRC.Models.Knn_model import train_knn_model
from SRC.Models_Evaluation.Evaluate_model import evaluate_model




def run_data_science_pipeline(train_path, test_path):
    
    print("---Loading data...")
    train, test = load_data(train_path, test_path)

    print("---Cleaning data...")
    train = clean_data(train)
    test = clean_data(test)

    print("---Performing EDA...")
    perform_eda(train)

    print("---Plotting histograms...")
    plot_histograms(train)

    print("---Training KNN model...")
    pipeline, X_val, y_val, submission = train_knn_model(train, test)

    print("---Evaluating KNN model...")
    evaluate_model(pipeline, X_val, y_val)

    return submission
run_data_science_pipeline('Data/Raw/train.csv', 'Data/Raw/test.csv')
