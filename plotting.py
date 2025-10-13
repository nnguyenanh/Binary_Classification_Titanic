# THIRD-PARTY LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
    # from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


# LOCAL LIBRARIES
from preprocessing import generatePath

# GLOBAL CONSTANTS
NAME_DT = 'DECISION TREE'
NAME_RF = 'RANDOM FORESTS'
NAME_GB = 'GRADIENT BOOSTING'
NAME_LIST = [NAME_DT, NAME_RF, NAME_GB]

PATH_STATISTICS = 'analysis/'


'''
STATISTICAL PLOTTING FUNCTIONS:

a. Correlation Heatmap
b. Features Importance
c. Confusion Matrix
d. ROC, AUC Curves
e. Precision-Recall Curves
'''

# a. CORRELATION HEATMAP 
def plotCorrelationHeatmap(df: pd.DataFrame, engineered_features: list[str]):
    
    plt.figure(figsize=(8, 8))
    correlation = df[engineered_features + ['Survived']].corr()

    im = plt.imshow(correlation, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(im)

    # Tick labels
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha='right', fontsize=15)
    plt.yticks(range(len(correlation.columns)), correlation.columns, fontsize=15)

    # Fill heatmap
    for pos_y in range(len(correlation.columns)):
        for pos_x in range(len(correlation.columns)):
            plt.text(
            pos_x, pos_y,
            f'{correlation.iloc[pos_y, pos_x]:.2f}',
            ha='center', va='center', fontsize=15
            )

    plt.title('Titanic Correlation Heatmap\n', fontsize=25)
 
    plt.tight_layout()
    plt.savefig(generatePath(PATH_STATISTICS + 'a_correlation_heatmap'), dpi=500, bbox_inches='tight')
    plt.close()


# b. FEATURES IMPORTANCE
def plotFeaturesImportance(MODEL_LIST: list[DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier], 
                           engineered_features: list[str], 
                           X_train, X_test, y_train, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    ticks = np.arange(0, 1.1, 0.1)
    labels = [f'{t:.1f}' for t in ticks]
    for model, name, ax in zip(MODEL_LIST, NAME_LIST, axes):
        importances = model.feature_importances_
        ax: Axes = ax
        ax.barh(engineered_features, importances, height=0.4)
        
        ax.set_xticks(ticks, labels, fontsize=15)
        ax.set_yticks(range(len(engineered_features)), engineered_features, fontsize=15)
        ax.set_xlabel('Importance')
        ax.set_title(f'{name}\n\n' +
                     f'Train set score: {model.score(X_train, y_train):.4f}\n' +
                     f'Test set score: {model.score(X_test , y_test):.4f}',
                     fontsize=20)
        
    fig.suptitle('\nFeature Importance Comparison\n', fontsize=30)
    plt.tight_layout()
    
    plt.savefig(generatePath(PATH_STATISTICS + 'b_features_importance'))


# c. CONFUSION MATRIX
def plotConfusionMatrix(MODEL_LIST: list[DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier],
                        Y_PRED_LIST: np.ndarray,
                        X_train, X_test, y_train, y_test):
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    
    for model, y_pred, name, ax in zip(MODEL_LIST, Y_PRED_LIST,NAME_LIST, axes):
        ax: Axes = ax
        cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
        ax.imshow(cm, cmap='Oranges')
        
        # Fill matrix
        ax.set_xticks([0, 1], ['Survived (1)', 'Dead (0)'], rotation=45, fontsize=25)
        ax.set_yticks([0, 1], ['Survived (1)', 'Dead (0)'], fontsize=25)

        for pos_y in range(cm.shape[0]):
            for pos_x in range(cm.shape[1]):
                ax.text(pos_y, pos_x, cm[pos_y, pos_x],
                        ha='center', va='center', fontsize=30)

        ax.set_title(f'{name}\n\n' +
                     f'Train score: {model.score(X_train, y_train):.4f}\n' +
                     f'Test score: {model.score(X_test, y_test):.4f}',
                     fontsize=25)
    
    fig.suptitle('\nConfusion Matrix Comparison', fontsize=40)
    plt.tight_layout()
    plt.savefig(generatePath(PATH_STATISTICS + 'c_confusion_matrix'))


# d. ROC, AUC CURVES
def plotROCCurves(MODEL_LIST: list[DecisionTreeClassifier | RandomForestClassifier | GradientBoostingClassifier],
                  X_test, y_test):
    
    plt.figure(figsize=(16, 12))

    for model, name in zip(MODEL_LIST, NAME_LIST):
        # Predicted probability for class 1
        y_prob = model.predict_proba(X_test)[:, 1]

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, marker='o', markersize=5,
                 label=f'{name} (AUC = {roc_auc:.3f})')

    # Random guess baseline
    plt.plot([0, 1], [0, 1], '--', label='Random (AUC = 0.5)')
    ticks = np.arange(0, 1.0, 0.11)
    labels = [f'{t:.1f}' for t in ticks]
    plt.xticks(ticks, labels, fontsize=20)
    plt.yticks(ticks, labels, fontsize=20)
    plt.xlabel('False Positive Rate', fontsize=25)
    plt.ylabel('True Positive Rate', fontsize=25)
    plt.title('\nROC Curve Comparison\n', fontsize=40)
    plt.legend(loc='lower right', fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(generatePath(PATH_STATISTICS + 'd_roc_curves'),
                dpi=500, bbox_inches='tight')
    plt.close()
    