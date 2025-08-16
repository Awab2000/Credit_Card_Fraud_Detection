from sklearn.metrics import classification_report, average_precision_score
from credit_fraud_utils_data import load_data, from_df_to_np, log_transform
import pickle
import time

def evaluate_model(model, x, y, threshold=0.5, is_hard=False):

    if not is_hard:
        y_proba = model.predict_proba(x)[:,1]
        y_pred = (y_proba >= threshold).astype(int)
    else:
        start = time.time()
        y_pred = model.predict(x)

        print(f"Execution time: {time.time() - start:.0f} seconds")

    report_dict = classification_report(y, y_pred, output_dict=True)

    report_str = classification_report(y, y_pred)

    ap = average_precision_score(y, y_pred)

    return report_dict, report_str, ap



if __name__ == '__main__':

    file_name = 'best_model/knn_model.pkl'
    with open(file_name, 'rb') as file:
        model = pickle.load(file)

    file_name = 'best_model/knn_processor.pkl'
    with open(file_name, 'rb') as file:
        processor = pickle.load(file)

    file_name = 'best_model/pca.pkl'
    with open(file_name, 'rb') as file:
        pca = pickle.load(file)

    df_test = load_data('Data/test.csv')

    X_test, y_test = from_df_to_np(df_test)

    X_test = pca.transform(X_test)

    X_test = processor.transform(X_test)

    X_test, _, _ = log_transform(X_test)


    test_report_dict, test_report_str, ap_test = evaluate_model(model['model'], X_test, y_test, model['threshold'], is_hard=True)

    print(f'Threshold = {model["threshold"]}')
    print(f'Test Report\n{test_report_str}')
    print(f'Test Average Precision: {ap_test}')
    # F1-Score = 0.81, Test Average Precision: 0.6544468760514747