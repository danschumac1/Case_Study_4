"""
Created on 04/09/2024

@author: Dan
"""

#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
# Standard import
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, root_mean_squared_error, r2_score, accuracy_score, make_scorer, mean_absolute_error
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

# Statistical imports
import statsmodels.api as sm
from statsmodels.formula.api import ols
from tqdm import tqdm

# Other imports
from ISLP import load_data
from datetime import datetime

#endregion DATA LOADING
#region # DATA LOADING
# =============================================================================
# IMPORT DATA
# =============================================================================

df = pd.read_csv('./data/CaseStudy4.csv')
df.drop(columns='Unnamed: 0', inplace=True)

X = df[['acq_exp','acq_exp_sq','industry','revenue','revenue','employees']]
y = df['acquisition']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# start a results dictionary
results_dict = {}

#endregion
#region # LOGISTIC REGRESSION
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================
log_clf = LogisticRegression()
log_clf.fit(X_train,y_train,)
log_preds = log_clf.predict(X_test)
log_acc = accuracy_score(y_test, log_preds)
results_dict['log'] = log_acc
print(log_acc)

#endregion
#region # DECISION TREE
# =============================================================================
# DECISION TREE
# =============================================================================
dt_clf = tree.DecisionTreeClassifier()
dt_clf.fit(X_train,y_train)
dt_preds = dt_clf.predict(X_test)
dt_acc = accuracy_score(y_test, dt_preds)
results_dict['dt'] = dt_acc
print(dt_acc)


#endregion
#region # RANDOM FOREST
# =============================================================================
# RANDOM FOREST 
# =============================================================================
# get features needed
max_features = X_train.shape[1]
tree_count = 1000

# set up model
rf_clf = RandomForestClassifier(
    max_features=max_features,
    random_state=27,
    n_estimators=tree_count
)

# fit model
rf_clf.fit(X_train, y_train)
# make predictions
rf_preds = rf_clf.predict(X_test)

# get acc
rf_acc = accuracy_score(y_test, rf_preds)
results_dict['rf'] = rf_acc
print(rf_acc)


# =============================================================================
# OPTIMIZING RANDOM FOREST
# =============================================================================

# NOW THAT WE KNOW THAT RANDOM FOREST IS BEST
# REFIT WITH GRIDSEARCH OPTIMIZATION
param_grid = {
    'n_estimators': [100, 500, 1000],  # Different numbers of trees
    'max_features': ['auto', 'sqrt', 'log2', None],  # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30],  # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required at each leaf node
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=27),
    param_grid=param_grid,
    cv=5,  # Number of folds in cross-validation
    scoring='accuracy',  # Metric to evaluate the models
    verbose=1,  # Show progress messages
    n_jobs=-1  # Use all available cores
)

grid_search.fit(X_train, y_train)

# Assuming grid_search is a scikit-learn GridSearchCV object
grid_search_preds = grid_search.predict(X_test)

# get results
optim_rf = accuracy_score(y_test, grid_search_preds)
results_dict['optim_rf'] = optim_rf
print(optim_rf)

print(results_dict)
# =============================================================================
# FIT BEST MODEL TO ENTIRE DATASET
# =============================================================================
whole_preds = grid_search.predict(X)
df['preds'] = whole_preds

#endregion
#region # VARIABLE IMPORTANCE PLOT
# =============================================================================
# VARIABLE IMPORTANCE PLOT
# =============================================================================
# Can't use non parametric model selection for parametric
# maybe 1 var in np is important, but not in para
# AIC is the way of capturing 
# significance doesn't mean important variable
# sig just mean is there a slope.
# there are interaction effects that compounds 
# AIC captures info bobtained by a variable.


# Create a DataFrame with feature names and their importance scores
plot_df = pd.DataFrame({
    'feature': X_train.columns,  # Use the column names from the DataFrame
    'importance': rf_clf.feature_importances_  # Feature importances from the RandomForestRegressor
})

# Plotting feature importances
plt.figure(figsize=(10, 10))
sns.barplot(x='importance', y='feature', data=plot_df.sort_values('importance', ascending=False))
plt.xticks(rotation=90)
plt.title('Feature Imortance')
plt.show()

#endregion
#region # RANDON FOREST REGRESSION
# =============================================================================
# RANDON FOREST REGRESSION
# =============================================================================


# cross val to find the best Hyper Parameters. @$@ @$@

# these are customers who we are predicting
df_pred_customer =df[df['preds'] == 1]
# train test test split / 80 20
X = df_pred_customer.drop(['duration','customer','acquisition'], axis = 1)
y = df_pred_customer['duration']

# CROSS VALIDATION
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

# FIT THE MODEL
rf_reg = RandomForestRegressor()
grid_search = GridSearchCV(estimator=rf_reg, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)


    
best_rf_reg = grid_search.best_estimator_

# Assuming the best parameters have been found:
best_params = grid_search.best_params_

# Instantiate RandomForestRegressor with OOB score enabled using best parameters
rf_reg_oob = RandomForestRegressor(
    n_estimators=best_params['n_estimators'],
    max_features=best_params['max_features'],
    min_samples_split=best_params['min_samples_split'],
    min_samples_leaf=best_params['min_samples_leaf'],
    oob_score=True,  # Enable OOB scoring
    random_state=42  # For reproducibility
)

# Assuming 'X' and 'y' are your features and target variable from the train set, and 'X_test', 'y_test' from the test set
# Model fitting
rf_reg_oob.fit(X, y)

# Predictions
y_pred = rf_reg_oob.predict(X)

# Calculating metrics

mse = root_mean_squared_error(y, y_pred)
rmse = root_mean_squared_error(y, y_pred)  # Pass squared=False to get the RMSE
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)
