# https://www.kaggle.com/code/sinakhorami/titanic-best-working-classifier
# https://www.kaggle.com/competitions/titanic/data?select=test.csv

# BUILD-IN LIBRARIES

# THIRD-PARTY LIBRARIES
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# LOCAL LIBRARIES
from preprocessing import (
    RANDOM_STATE_DATA,
    loadPreprocessedTitanicData, 
    generatePath
)
from training import (
    PATH_MODEL,
    NAME_DT,
    NAME_RF,
    NAME_GB,
    load_model
)

from plotting import (
    plotCorrelationHeatmap,
    plotFeaturesImportance,
    plotConfusionMatrix,
    plotROCCurves
)


def main():
    
    '''
    LOAD PREPROCESSED DATA
    '''
    df, engineered_features = loadPreprocessedTitanicData()

    X = df.drop(columns='Survived')
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE_DATA)
    
    '''
    TRAIN MODELS
    '''
    # a. Decision Tree
    decision_tree: DecisionTreeClassifier = load_model(generatePath(PATH_MODEL + NAME_DT))
    
    # b. Random Forests
    random_forests: RandomForestClassifier = load_model(generatePath(PATH_MODEL + NAME_RF))
    
    # c. Gradient Boost
    gradient_boost: GradientBoostingClassifier = load_model(generatePath(PATH_MODEL + NAME_GB))
    
    '''
    PREDICTIONS
    '''
    y_pred_dt = decision_tree.predict(X_test)
    y_pred_rf = random_forests.predict(X_test)
    y_pred_gb = gradient_boost.predict(X_test)

    
    '''
    PLOT MODELS STATISTIC (3 IN 1 FIGURES) 
    '''
    # General purpose variables
    model_list = [decision_tree, random_forests, gradient_boost]
    y_pred_list = [y_pred_dt, y_pred_rf, y_pred_gb]
    
    # a. Correlation Heatmap
    plotCorrelationHeatmap(df, engineered_features)
    
    # b. Features Importance Comparision
    plotFeaturesImportance(model_list, engineered_features,
                           X_train, X_test, y_train, y_test)
    
    # c. Confusion Matrix Comparision
    plotConfusionMatrix(model_list, y_pred_list,
                        X_train, X_test, y_train, y_test)
    
    # d. ROC Curves with AUC
    plotROCCurves(model_list, X_test, y_test)



if __name__ == '__main__':
    # test()
    main()
    print('Done.')
