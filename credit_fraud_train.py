import argparse
import json
import os
import pickle
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from credit_fraud_utils_data import load_data, remove_duplicates, under_sample, log_transform, over_sample
from credit_fraud_utils_data import from_df_to_np, transform_train_val, poly_process, under_over_sample
from credit_fraud_utils_eval import evaluate_model
from collections import Counter


def save_model_metadata(model_name, model_hyperparameters, threshold, train_report, val_report, ap_train, ap_val, selected_features,
                        poly_degree, processor, is_log_transformed, train_rows,
                        val_rows, imbalance_techniques):
    json_file = "model_history.json"

    metrics = {
        "0 train F1-Score": train_report['0']['f1-score'],
        "1 train F1-Score": train_report['1']['f1-score'],
        "Train macro average F1-Score": train_report['macro avg']['f1-score'],
        "Train PR-AUC": ap_train,
        "0 val F1-Score": val_report['0']['f1-score'],
        "1 val F1-Score": val_report['1']['f1-score'],
        "Val macro average F1-Score": val_report['macro avg']['f1-score'],
        "Val PR-AUC": ap_val,
    }

    if os.path.exists(json_file):
        with open(json_file, 'r') as file:
            data = json.load(file)
    else:
        data = {"models" : []}

    new_model_entry = {
        "count": len(data["models"]),
        "model_name": model_name,
        "model_hyperparameters": model_hyperparameters,
        "Threshold": threshold,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "selected features": selected_features,
        "polynomial degree": poly_degree,
        "processor": processor,
        "log transformed?": is_log_transformed,
        "train # rows": train_rows,
        "val # rows": val_rows,
        "imbalance techniques": imbalance_techniques
    }


    data["models"].append(new_model_entry)

    with open(json_file, 'w') as file:
        json.dump(data, file, indent=2)

    print(f"model {model_name} history updated succefully")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Credit Card Fraud Detection')

    parser.add_argument('--model_choice', type=int, default=4,
                        help='1 For LogisticRegression, '
                             '2 For RandomForestClassifier, '
                             '3 For NeuralNetwork, '
                             '4 For VotingClassifier')

    parser.add_argument('--train_dataset', type=str, default='Data/train.csv')

    parser.add_argument('--val_dataset', type=str, default='Data/val.csv')

    parser.add_argument('--processing', type=int, default=1,
                        help='0 For no processing, '
                             '1 For min-max scaling, '
                             '2 For standarizing')

    parser.add_argument('--log_transform', type=int, default=1,
                        help='0 For no log transformation, '
                             '1 For log transform, ')

    parser.add_argument('--poly_degree', type=int, default=1, help='Degree is the value of the argument')

    args = parser.parse_args()

    df_train = load_data(args.train_dataset)

    df_val = load_data(args.val_dataset)

    df_train = remove_duplicates(df_train)
    df_val = remove_duplicates(df_val)

    selected_columns = df_train.columns.tolist()[:-1]

    X_train, y_train = from_df_to_np(df_train)

    X_val, y_val = from_df_to_np(df_val)

    X_train, X_val, processor_type = transform_train_val(X_train, X_val, args.processing)

    # X_train, X_val = poly_process(X_train, X_val, args.poly_degree)

    if args.log_transform:
        X_train, X_val, _ = log_transform(X_train, X_val)

    # X_train, y_train = under_sample(X_train, y_train)

    # X_train, y_train = over_sample(X_train, y_train)

    X_train_ou, y_train_ou = under_over_sample(X_train, y_train)

    train_rows = X_train.shape[0]
    val_rows = X_val.shape[0]


    if args.processing == 0:
        processor = "No processing"
    elif args.processing == 1:
        processor = "Min/Max scaler"
    else:
        processor = "Standard Scaler"

    ws = [0.01, 0.05, 0.1, 0.5, 1]

    counter = Counter(y_train)
    ir = counter[1] / counter[0]
    # ws = [ir]

    depths = [3, 7, 10]
    estimators = [5, 20, 40, 70, 100]

    hidden_layer_sizes = [25, 50, 75, 100]
    es = [False, True]
    lrs = [0.1, 0.01, 0.001, 0.0001]

    voting_weights = [0.25, 0.5, 1.0, 1.5, 2]
    voting_type = [True, False]

    threshold = 0.2

    if args.model_choice == 1:
        model_name = "LogisticRegression"
        model = LogisticRegression(class_weight={0:0.5,1:1}, warm_start=True)
    elif args.model_choice == 2:
        model_name = "RandomForestClassifier"
        model = RandomForestClassifier(class_weight={0:0.5,1:1}, max_depth=10, n_estimators=50, warm_start=True)
    elif args.model_choice == 3:
        model_name = "NeuralNetwork"
        model = MLPClassifier(hidden_layer_sizes=(25,), learning_rate_init=0.1, warm_start=True, early_stopping=False, random_state=41)
    else:   # Voting Classifier
        model_name = "VotingClassifier"
        model1 = RandomForestClassifier(class_weight={0: 0.5, 1: 1}, max_depth=7, n_estimators=5,
                                        warm_start=True)
        model2 = LogisticRegression()
        model1.fit(X_train_ou, y_train_ou)
        model = VotingClassifier(estimators=[('random_forest', model1), ('logistic_regression', model2) ], voting='soft', weights=[1.5,1.0])

    # model.fit(X_train_ou, y_train_ou)

    model.fit(X_train, y_train)

    # model_dict = {
    #     "model": model,
    #     "threshold": threshold,
    #     "model_name": model_name
    # }
    #
    # root_dir = 'best_model'
    # with open(os.path.join(root_dir, 'model.pkl'), 'wb') as file:
    #     pickle.dump(model_dict, file)
    #
    # root_dir = 'best_model'
    # with open(os.path.join(root_dir, 'processor.pkl'), 'wb') as file:
    #     pickle.dump(processor_type, file)

    train_report_dict, train_report_str, ap_train = evaluate_model(model, X_train, y_train, threshold, False)

    val_report_dict, val_report_str, ap_val = evaluate_model(model, X_val, y_val, threshold, False)

    print(f'Threshold = {threshold}')
    print(f'Train Report\n{train_report_str}')
    print(f'Train Average Precision: {ap_train}')
    print(f'Val Report\n{val_report_str}')
    print(f'Val Average Precision: {ap_val}')

    # save_model_metadata(model_name, f"estimators=[('random_forest', model1=RandomForestClassifier(class_weight=(0:0.5,1:1), max_depth=7, n_estimators=5, warm_start=True)), ('logistic_regression', model2=LogisticRegression())], voting=soft}, weights=[{vw1},{vw2}]", threshold, train_report_dict, val_report_dict, ap_train, ap_val, selected_columns,
    #                     args.poly_degree, processor, args.log_transform, train_rows, val_rows, "Oversampling SMOTE(random_state=1, sampling_strategy={1: majority_size / 100}, k_neighbors=3) + Undersampling RandomUnderSampler(random_state=1, sampling_strategy={0: minority_size * 100})") # Oversampling SMOTE(random_state=1, sampling_strategy={1: majority_size / 100}, k_neighbors=3) + Undersampling RandomUnderSampler(random_state=1, sampling_strategy={0: minority_size * 100})