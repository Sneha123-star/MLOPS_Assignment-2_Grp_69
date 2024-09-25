# Importing necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sweetviz as sv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tpot import TPOTClassifier
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

def load_data(path):
    #Load the wineequality-red.csv data
    # df1 = pd.read_csv('winequality-red.csv',sep=";")
    df = pd.read_csv(path,sep=";")
    # df=pd.concat([df1,df2],axis=1)
    return df

def auto_eda(df):
    # AutoEDA using Sweetviz
    # Generate Sweetviz report
    report = sv.analyze(df)
    # Save the report as HTML
    report.show_html('wine_quality_sweetviz_report.html')  # Save report


def split_data(df):
    # Separate features and target
    X = df.drop('quality', axis=1)
    y = df['quality']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, y_train, y_test):
    # scaling the input data
    scaler_x = MinMaxScaler()
    scaler_x.fit(X_train)
    scaled_X_train = scaler_x.transform(X_train)
    scaled_X_test = scaler_x.transform(X_test)
    # scaling the output data
    scaler_y = MinMaxScaler()
    scaler_y.fit(y_train)
    scaled_y_train = scaler_y.transform(y_train)
    scaled_y_test = scaler_y.transform(y_test)

    return scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test

def train_model(scaled_X_train):

    # Feature Engineering
    df['wine_class'] = df['quality'].apply(lambda x: 1 if x >= 7 else (0 ))

    # Model Selection and Hyperparameter Tuning with TPOT
    # Use TPOT for AutoML model selection and hyperparameter tuning
    tpot = TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=12)

    # Fit the TPOT model on training data
    tpot.fit(X_train_scaled, scaled_y_train)


def main(parameters):

    path = "'winequality-white.csv'"
    df = load_data(path)

    X_train, X_test, y_train, y_test = split_data(df)

    scaled_X_train, scaled_X_test, scaled_y_train, scaled_y_test = scale_data(X_train, X_test, y_train, y_test)

    # Creating parameter dictionary for RF
    params_dict_rf = {
        "verbosity" : 2, 
        "generations" : 5, 
        "population_size" : 20,
    }

    train_model(scaled_X_train, scaled_y_train, scaled_y_train, params_dict_rf)





# Evaluate on test data
y_pred = tpot.predict(X_test_scaled)

# Save preprocessed data and best model
processed_data = {
    'X_train': X_train_scaled,
    'X_test': X_test_scaled,
    'y_train': y_train,
    'y_test': y_test
}

# Save the best model and preprocessed data to pickle files
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

tpot.export('best_model_tpot.pkl')  # Save the best model pipeline as Python code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--n_estimators", type=int, default=1000)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=5)
    parser.add_argument("--kernel", type=str, default='linear')
    parser.add_argument("--C", type=float, default=1)
    parser.add_argument("--gamma", type=str, default='scale')
    parser.add_argument("--epsilon", type=float, default=0.2)

    args = parser.parse_args()

    with mlflow.start_run():
        # main(args.max_depth, args.min_samples_split)  # DT
        main(args.n_estimators, args.max_depth, args.min_samples_split) # RF
        # main(args.kernel, args.C, args.gamma, args.epsilon) # SVM


