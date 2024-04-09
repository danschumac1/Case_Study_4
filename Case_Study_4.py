"""
Created on 04/09/2024

@author: Dan
"""

#region # IMPORTS
# =============================================================================
# IMPORTS
# =============================================================================
import pandas as pd
# Standard imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning imports
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score

# Statistical imports
import statsmodels.api as sm
from statsmodels.formula.api import ols

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


#endregion
#region # RANDOM FOREST
# =============================================================================
# RANDOM FOREST 
# =============================================================================

# FIT RANDOM FOREST
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

# get mse
rf_mse = mean_squared_error(y_test, rf_preds)

# NOW THAT WE KNOW THAT RANDOM FOREST IS BEST
rf_clf.fit(X, y)
rf_final_preds = rf_clf.predict(X)
df['preds'] = rf_final_preds

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
#region # LOGISTIC REGRESSION
# =============================================================================
# LOGISTIC REGRESSION
# =============================================================================

#endregion
#region # DESCISION TREE
# =============================================================================
# DESCISION TREE
# =============================================================================

#endregion
#region # REGRESSION
# =============================================================================
# REGRESSION
# =============================================================================
df.columns
X = df.drop(['duration','customer','acquisition'])
y = df['duration']
rf_reg = RandomForestRegressor()
rf_reg.fit(X,y)
reg_preds = rf_reg.predict(X)

df['predicted_duration'] = reg_preds

# these are customers who we are predicting
df_pred_customer =df[df['preds'] == 1]

# these are the predicted customers predicted duration
df_pred_customer['predicted_duration']