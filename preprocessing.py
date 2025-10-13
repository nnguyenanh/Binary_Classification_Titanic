# https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier
# https://www.kaggle.com/competitions/titanic/data?select=test.csv

# BUILD-IN LIBRARIES
from pathlib import Path

# THIRD-PARTY LIBRARIES
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# GLOBAL CONSTANTS
PATH_DATA = 'data_folder/'
PATH_DATA_RAW = 'data_folder/titanic_data.csv'
PATH_DATA_PROCESSED = 'data_folder/titanic_data_processed.csv'
RANDOM_STATE_DATA = 66
RANDOM_STATE_TRAIN = 2025
ENGINEERED_FEATURES = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize']

'''
PREPROCESS RAW DATA:

a. Preprocess Titanic Data
c. Load preprocessed Titanic Data
b. Generate Absolute Path
'''

# a. Preprocess Titanic Data
def preprocessedTitanicData():

    # Make a copy to work on
    original_df = pd.read_csv(generatePath(PATH_DATA_RAW))
    preprocessed_df = original_df.copy()
    
    # 1.
    # Drop all irrelevant features 1
    features_drop = [
        'PassengerId',
        'Name', 
        'Ticket',
        'Cabin', 
        'Embarked', 
    ]
    preprocessed_df.drop(columns=features_drop, inplace=True)
    
    # 2.
    # Encode Sex: Male to 1, Female to 0
    preprocessed_df['Sex'] = preprocessed_df['Sex'].map({'male': 1, 'female': 0})
    
    # 3. 
    # Handle missing ages by Random Forest
    # Create Series of filled and empty ages data frames
    filled_age = preprocessed_df[preprocessed_df['Age'].notnull()]
    empty_age = preprocessed_df[preprocessed_df['Age'].isnull()]
    
    # Create X_train and y_train
    features_used = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare']
    X_train = filled_age[features_used]
    y_train = filled_age['Age']
    
    # Train model by filled_age dataframe
    model = RandomForestRegressor(
        n_estimators=100, max_depth=5,
        random_state=66, 
    )
    model.fit(X_train, y_train)
    
    # Predict empty ages
    X_pred = empty_age[features_used]
    y_pred = model.predict(X_pred)    
    
    y_pred = np.floor(y_pred).astype(int) 
    
    preprocessed_df.loc[preprocessed_df['Age'].isnull(), 'Age'] = y_pred
    
    # 4. 
    # Features Engineering
    preprocessed_df['FamilySize'] = preprocessed_df['SibSp'] + preprocessed_df['Parch'] + 1

    # 5.
    # Drop all irrelevant features 2
    features_drop = [
         'SibSp', 
         'Parch',
    ]
    preprocessed_df.drop(columns=features_drop, inplace=True)
    # Save preprocessed data
    preprocessed_df.to_csv(generatePath(PATH_DATA_PROCESSED), index=False)
    print('Data processing finish.')

# b. Load preprocess Titanic Data
def loadPreprocessedTitanicData():

    # Make a copy to work on
    preprocessed_df = pd.read_csv(generatePath(PATH_DATA_PROCESSED))
    
    return preprocessed_df, list(preprocessed_df.columns.drop('Survived'))

# c. Generate Absolute Path
def generatePath(file_name: str) -> str:
    # Get current path and concatnate with parameter
    base = Path(__file__).resolve().parent          
    csv_path = base / file_name
    
    return str(csv_path)

if __name__ == '__main__':
    preprocessedTitanicData()
