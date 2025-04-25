import json


json_file = "model_history.json"

with open(json_file, 'r') as file:
    data = json.load(file)


model_score = -1
best_model = {}

for dct in data["models"]:
    if dct["model_name"] == "VotingClassifier":
        if dct["metrics"]["1 val F1-Score"] > model_score:
            best_model = dct
            model_score = dct["metrics"]["1 val F1-Score"]


print(best_model, "\n\n")




# {'count': 36, 'model_name': 'LogisticRegression', 'model_hyperparameters': 'Default', 'Threshold': 0.5, 'date': '2025-04-04 17:16:55', 'metrics': {'0 train F1-Score': 0.9951710261569416, '1 train F1-Score': 0.8651685393258427, 'Train macro average F1-Score': 0.9301697827413922, 'Train PR-AUC': 0.7666342096721843, '0 val F1-Score': 0.9997095658449434, '1 val F1-Score': 0.8092485549132948, 'Val macro average F1-Score': 0.9044790603791191, 'Val PR-AUC': 0.6557646425043672}, 'selected features': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'], 'polynomial degree': 2, 'processor': 'Min/Max scaler', 'log transformed?': 1, 'train # rows': 7722, 'val # rows': 56898, 'imbalance techniques': 'Undersampling, sampling_strategy={0: 25 * minority_size}'}

# {'count': 620, 'model_name': 'RandomForestClassifier', 'model_hyperparameters': 'class_weight=(0:0.5,1:1), warm_start=True, max_depth=7, n_estimators=5', 'Threshold': 0.5, 'date': '2025-04-08 01:00:18', 'metrics': {'0 train F1-Score': 0.9996620102454982, '1 train F1-Score': 0.816, 'Train macro average F1-Score': 0.9078310051227491, 'Train PR-AUC': 0.6677445790654869, '0 val F1-Score': 0.9997623636891717, '1 val F1-Score': 0.847457627118644, 'Val macro average F1-Score': 0.9236099954039079, 'Val PR-AUC': 0.7184534087861687}, 'selected features': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'], 'polynomial degree': 1, 'processor': 'Min/Max scaler', 'log transformed?': 1, 'train # rows': 170436, 'val # rows': 56898, 'imbalance techniques': 'Oversampling SMOTE(random_state=1, sampling_strategy={1: majority_size / 100}, k_neighbors=3) + Undersampling RandomUnderSampler(random_state=1, sampling_strategy={0: minority_size * 100})'}

# {'count': 859, 'model_name': 'NeuralNetwork', 'model_hyperparameters': 'hidden_layer_sizes=(25,), learning_rate_init=0.01, warm_start=True, early_stopping=False, random_state=41', 'Threshold': 0.1, 'date': '2025-04-08 02:26:26', 'metrics': {'0 train F1-Score': 0.9996885137994264, '1 train F1-Score': 0.8133802816901409, 'Train macro average F1-Score': 0.9065343977447836, 'Train PR-AUC': 0.6633638718982416, '0 val F1-Score': 0.999735975921004, '1 val F1-Score': 0.8235294117647058, 'Val macro average F1-Score': 0.911632693842855, 'Val PR-AUC': 0.6800398540454963}, 'selected features': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'], 'polynomial degree': 1, 'processor': 'Min/Max scaler', 'log transformed?': 1, 'train # rows': 170436, 'val # rows': 56898, 'imbalance techniques': 'Oversampling SMOTE(random_state=1, sampling_strategy={1: majority_size / 100}, k_neighbors=3) + Undersampling RandomUnderSampler(random_state=1, sampling_strategy={0: minority_size * 100})'}

# {'count': 1558, 'model_name': 'VotingClassifier', 'model_hyperparameters': "estimators=[('random_forest', model1=RandomForestClassifier(class_weight=(0:0.5,1:1), max_depth=7, n_estimators=5, warm_start=True)), ('logistic_regression', model2=LogisticRegression())], voting=soft, weights=[1.5,1.0]", 'Threshold': 0.2, 'date': '2025-04-14 20:36:56', 'metrics': {'0 train F1-Score': 0.9997443439778314, '1 train F1-Score': 0.8476357267950964, 'Train macro average F1-Score': 0.9236900353864639, 'Train PR-AUC': 0.7199766623064929, '0 val F1-Score': 0.9997799702519781, '1 val F1-Score': 0.8571428571428571, 'Val macro average F1-Score': 0.9284614136974176, 'Val PR-AUC': 0.7351559054011293}, 'selected features': ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'], 'polynomial degree': 1, 'processor': 'Min/Max scaler', 'log transformed?': 1, 'train # rows': 170436, 'val # rows': 56898, 'imbalance techniques': 'Oversampling SMOTE(random_state=1, sampling_strategy={1: majority_size / 100}, k_neighbors=3) + Undersampling RandomUnderSampler(random_state=1, sampling_strategy={0: minority_size * 100})'}


