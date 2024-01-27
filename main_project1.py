#%% Setup
import numpy as np
import pandas as pd
import itertools
from scipy.stats import shapiro
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    confusion_matrix, 
    make_scorer,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None) 

from lazypredict.Supervised import LazyClassifier

from functions import (
    create_table_categorical,
    create_graph_categorical,
    create_table_numeric_continuous,
    create_graph_numeric_continuous,
    bivariate,
    create_table_bivariate_summary,
    create_table_bivariate_html,
    create_graph_bivariate_html,
    create_graph_h_bivariate_html,
    binning_to_model,
    create_ks_table_for_logistic_regression,
    calculate_CramersV,
    calculate_CramersV2,
    show_df
    )

#%%
# Read files
file_name   = "Bank Customer Churn Prediction"
df_original = pd.read_csv(f"./source/{file_name}.csv")
df          = df_original.copy()

numerical_variables = [
    "products_number",
    "tenure",
    "credit_score",
    "age",
    "estimated_salary"
    ]

categorical_variables = [
    "country", 
    "gender", 
    "credit_card",
    "active_member",
    "balance",
    ]

target_variable = 'churn'

"""
lst = numerical_variables + categorical_variables 

cols_combine = []
for i in range(5, len(lst) + 1):
    cols_combine += list(itertools.combinations(lst, i)) 

# Models
seed   = 100
df     = df[numerical_variables + categorical_variables + [target_variable]]
X_all  = df.drop(target_variable, axis=1).copy()
y      = df[target_variable].copy()
j      = 1
df_res = pd.DataFrame()

for m in cols_combine:
     
    print(j)
    j += 1
    
    # Select features
    X = X_all[list(m)]
    original_cols = X.columns

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    
    # ----------------------------------- Binnings --------------------------------
    df_train = pd.merge(X_train,
                        y_train, 
                        how         = "left", 
                        left_index  = True, 
                        right_index = True,
                        validate    = "one_to_one")
    
    df_test = pd.merge(X_test,
                       y_test, 
                       how         = "left", 
                       left_index  = True, 
                       right_index = True,
                       validate    = "one_to_one")
                
    
    df_train = binning_to_model(
        df_train,
        list(df_train.columns[df_train.columns.isin(numerical_variables)]),
        list(df_train.columns[df_train.columns.isin(categorical_variables)]),
        target_variable).reset_index(drop = True)
    
    df_test = binning_to_model(
        df_test,
        list(df_test.columns[df_test.columns.isin(numerical_variables)]),
        list(df_test.columns[df_test.columns.isin(categorical_variables)]),
        target_variable).reset_index(drop = True)
    
    X_train  = df_train.drop(target_variable, axis=1).copy()
    y_train  = df_train[target_variable].copy()
    
    X_test  = df_test.drop(target_variable, axis=1).copy()
    y_test  = df_test[target_variable].copy()
     
    # -----------------------------------------------------------------------------
    
    # One-hot Encoding
    enc = OneHotEncoder(handle_unknown='ignore', drop = 'first')
    enc.fit(X_train.astype(str))
    
    colnames = enc.get_feature_names_out()
    
    # train
    transformed = enc.transform(X_train.astype(str)).toarray()
    df_cat_vars = pd.DataFrame(columns=colnames, data=transformed)
    X_train = pd.concat([X_train, df_cat_vars], axis=1)
    
    # test
    transformed = enc.transform(X_test.astype(str)).toarray()
    df_cat_vars = pd.DataFrame(columns=colnames, data=transformed)
    X_test = pd.concat([X_test, df_cat_vars], axis=1)
    
    # Remocao das variáveis categoricas sem codificação
    X_train.drop(original_cols, axis=1, inplace = True)
    X_test.drop(original_cols, axis=1, inplace = True)
    
    # Models
    models = []
    models.append(("LogisticRegression",LogisticRegression()))

    for name,model in models:
        result = cross_val_score(model, X_train, y_train,  cv=5)
        
        clf = LogisticRegression()
        clf.fit(X_train, y_train)

        nos_train = create_ks_table_for_logistic_regression(clf, X_train, y_train)
        nos_test  = create_ks_table_for_logistic_regression(clf, X_test, y_test)
        
        
        df_aux = pd.DataFrame()
        df_aux["name"] = [name]
        df_aux["mean accuracy "] = [result.mean()]
        df_aux["std accuracy"]   = [result.std()]
        df_aux["predictors"]     = [m]
        df_aux["KS_train"]       = nos_train["KS"].unique()
        df_aux["KS_test"]        = nos_test["KS"].unique()
        
        df_res = pd.concat([df_res, df_aux])
    
df_res.reset_index(drop = True, inplace = True)
df_res.to_pickle("best_models.pkl")
"""

seed          = 100
X             = df[['products_number', 'country', 'gender', 'credit_card', 'active_member']].copy()
y             = df[target_variable].copy()
original_cols = X.columns

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

# ----------------------------------- Binnings --------------------------------
df_train = pd.merge(X_train,
                    y_train, 
                    how         = "left", 
                    left_index  = True, 
                    right_index = True,
                    validate    = "one_to_one")

df_test = pd.merge(X_test,
                   y_test, 
                   how         = "left", 
                   left_index  = True, 
                   right_index = True,
                   validate    = "one_to_one")
            

df_train = binning_to_model(
    df_train,
    list(df_train.columns[df_train.columns.isin(numerical_variables)]),
    list(df_train.columns[df_train.columns.isin(categorical_variables)]),
    target_variable).reset_index(drop = True)

df_test = binning_to_model(
    df_test,
    list(df_test.columns[df_test.columns.isin(numerical_variables)]),
    list(df_test.columns[df_test.columns.isin(categorical_variables)]),
    target_variable).reset_index(drop = True)

X_train  = df_train.drop(target_variable, axis=1).copy()
y_train  = df_train[target_variable].copy()

X_test  = df_test.drop(target_variable, axis=1).copy()
y_test  = df_test[target_variable].copy()
 
# -----------------------------------------------------------------------------

# One-hot Encoding
enc = OneHotEncoder(handle_unknown='ignore', drop = 'first')
enc.fit(X_train.astype(str))

colnames = enc.get_feature_names_out()

# train
transformed = enc.transform(X_train.astype(str)).toarray()
df_cat_vars = pd.DataFrame(columns=colnames, data=transformed)
X_train = pd.concat([X_train, df_cat_vars], axis=1)

# test
transformed = enc.transform(X_test.astype(str)).toarray()
df_cat_vars = pd.DataFrame(columns=colnames, data=transformed)
X_test = pd.concat([X_test, df_cat_vars], axis=1)

# Remocao das variáveis categoricas sem codificação
X_train.drop(original_cols, axis=1, inplace = True)
X_test.drop(original_cols, axis=1, inplace = True)

# Treino
clf = LogisticRegression()
clf.fit(X_train, y_train)

nos_train = create_ks_table_for_logistic_regression(clf, X_train, y_train)
nos_test  = create_ks_table_for_logistic_regression(clf, X_test, y_test)





   
    



 








