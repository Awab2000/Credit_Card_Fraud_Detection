# Credit_Card_Fraud_Detection

In this project, I developed a classification model to distinguish between fraudulent
and non-fraudulent credit card transactions made by European cardholders in
September 2013.

The dataset presents transactions that occurred in two days,
where we have 492 frauds out of 284,807 transactions. The dataset is highly
unbalanced, the positive class (frauds) account for 0.172% of all transactions. This
task was a [competition](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data) held by Kaggle.

The Project implements four different models using scikit-learn namely: VotingClassifier, RandomForestClassifier, LogisticRegression, MLPClassifier(NN). Acheiving an F1-Score of 0.80 and PR-AUC of 0.64 on test dataset.

### Installation

You need to install the following libraries: sklearn, numpy, pandas, matplotlib, seaborn, plotly, imblearn.

## Project Structure

```shall
credit Card Fraud Detection/
├── best_model/   
│   ├── model.pkl 
│   └── processor.pkl 
│         
├── Data/
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
|   |__ trainval.csv
|
├── credit_fraud_train.py
├── credit_fraud_utils_data.py
├── credit_fraud_utils_eval.py
├── Credit Card Fraud Detection EDA.ipynb
├── model_search.py
└── README.md
```

- `Data/`
  - `train.csv`: Training dataset.
  - `val.csv`: Validation dataset.
  - `test.csv`: Test dataset.
  - `trainval.csv`: Training + Validation dataset.
 
- `best_model`
  - `model.pkl`: best chosen model.
  - `processor.pkl`: Prcessor for transforming the data.
 
- `credit_fraud_train.py`: The script for training the models.
- `credit_fraud_utils_eval.py`: The script for evaluating models.
- `credit_fraud_utils_data.py`: The script for data processing and transformation.
- `model_search.py`: The script for searching on models from model_history.
- `model_history.json`: File contains all trained models and thier metadata.
- `Credit Card Fraud Detection EDA.ipynb`: Exploratory Data Analysis Notebook.

## Data and Modeling Pipeline

<div>
<img width="830" alt="Image" src="https://github.com/user-attachments/assets/24d54dae-d3e7-43be-a563-16c827634354" />
</div>


### How to configure this project for your own uses

I'd encourage you to clone and rename this project to use for your own puposes.
