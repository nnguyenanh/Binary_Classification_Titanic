# https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier
# https://www.kaggle.com/competitions/titanic/data?select=test.csv

# BUILD-IN LIBRARIES
import os

# THIRD-PARTY LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import joblib as jl
    # from sklearn.model_selection import GridSearchCV

# LOCAL LIBRARIES
from preprocessing import (
    loadPreprocessedTitanicData,
    generatePath,
    RANDOM_STATE_TRAIN,
    RANDOM_STATE_DATA
)

# CONSTANTS
PATH_MODEL = 'models/'
NAME_DT = 'decision_tree.pkl'
NAME_RF = 'random_forests.pkl'
NAME_GB = 'gradient_boost.pkl'
X_TEST = 'X_test.csv'
Y_TEST = 'y_test.csv' 

'''
MODELS TRAINING

a. Decision Tree
b. Random Forests
c. Gradient Boosting
d. Save model
e. Load model
'''

# a. DECISION TREE
def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    
    # Train model
    decision_tree = DecisionTreeClassifier(
        random_state=RANDOM_STATE_TRAIN,
        criterion='entropy',
        max_depth=5,
        min_samples_leaf=5,
        min_samples_split=3
    ).fit(X_train, y_train)

    # Use GridSearchCV to find best parameters
    # Set parameters grid
    # set_param_grid = {
    #     'max_depth': np.arange(2, 10, 1),
    #     'min_samples_split': np.arange(2, 10, 1),
    #     'min_samples_leaf': np.arange(2, 10, 1),
    #     'criterion': ['gini', 'entropy']
    # }
    
    # # Inititalize grid search with parameters
    # grid_search = GridSearchCV(
    #     estimator=decision_tree_model,
    #     param_grid=set_param_grid,
    #     cv=5, # 5-fold cross-validation
    # ).fit(X_train, y_train)
    
    # grid_search = findBestParametersDT(X_train, y_train, decision_tree_model)
    # best_criterion = grid_search.best_params_['criterion']
    # best_max_depth = grid_search.best_params_['max_depth']
    # best_min_samples_split = grid_search.best_params_['min_samples_split']
    # best_min_samples_leaf = grid_search.best_params_['min_samples_leaf']
    
    return decision_tree  


# b. RANDOM FORESTS
def train_random_forests(X_train, y_train) -> RandomForestClassifier:
    
    # Train model
    random_forests = RandomForestClassifier(
        random_state=RANDOM_STATE_TRAIN,
        criterion='entropy',
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        min_samples_split=10,
        n_jobs=-1
    ).fit(X_train, y_train)
    
    return random_forests  


# c. GRADIENT BOOSTING
def train_gradient_boosts(X_train, y_train) -> GradientBoostingClassifier:
    
    # Train model
    gradient_boost = GradientBoostingClassifier(
        random_state=RANDOM_STATE_TRAIN,
        # n_estimators=1000,
        # learning_rate=0.025,
        # max_depth=3,
        # min_samples_leaf=4,
        # min_samples_split=8,
    ).fit(X_train, y_train)

    return gradient_boost  
    
# d. TRAIN AND SAVE MODELS  
def trainsaveModels():
    
    '''
    LOAD PREPROCESSED DATA
    '''
    df, _ = loadPreprocessedTitanicData()

    X = df.drop(columns='Survived')
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE_DATA)


    '''
    TRAIN MODELS
    '''
    # a. Decision Tree
    decision_tree = train_decision_tree(X_train, y_train)
    save_model(decision_tree, generatePath(PATH_MODEL + NAME_DT))

    # b. Random Forests
    random_forests = train_random_forests(X_train, y_train)
    save_model(random_forests, generatePath(PATH_MODEL + NAME_RF))
    
    # c. Gradient Boost
    gradient_boosts = train_gradient_boosts(X_train, y_train)
    save_model(gradient_boosts, generatePath(PATH_MODEL + NAME_GB))
   
 
# e. SAVE MODEL TO FILE
def save_model(model, path):
    jl.dump(model, path)
    print(f'Model saved to {path}')
    
    
# f. LOAD MODEL FROM FILE
def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f'No model found at {path}')
    model = jl.load(path)
    print(f'Model loaded from {path}')
    return model


if __name__ == '__main__':
    trainsaveModels()