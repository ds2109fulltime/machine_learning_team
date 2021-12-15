import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score,\
                            roc_auc_score, roc_curve


def print_regress_metrics(y, y_pred):
    ''' 
    This function print the plot the R^2, MAE, MSE, RMSE and MAPE score of a regression model.

    Args:
        y (pandas.Series): The real target values.
        y_pred (pandas.Series): The target values predicted by the model.
    
    Returns:
        None

    '''

    print("R^2 score:", round(r2_score(y_pred, y), 4))
    print("MAE score:", round(mean_absolute_error(y_pred, y), 4))
    print("MSE score:", round(mean_squared_error(y_pred, y), 4))
    print("RMSE score:", round(np.sqrt(mean_squared_error(y_pred, y)), 4))
    y_array, y_pred_array = np.array(y), np.array(y_pred)
    mape = np.mean(np.abs((y_array - y_pred_array) / y_array)) * 100
    print(f'MAPE score: {round(mape, 4)} %')


def print_classif_metrics(y, y_pred):
    ''' 
    This function print the plot the accuracy, recall, precision, F1 score and AUC
        of a classification model.

    Args:
        y (pandas.Series): The real target values.
        y_pred (pandas.Series): The target values predicted by the model.
    
    Returns:
        None

    '''

    print(f'Accuracy score: {round(accuracy_score(y_pred, y), 3)} %')
    print(f'Recall score: {round(recall_score(y_pred, y), 3)} %')
    print(f'Precision score: {round(precision_score(y_pred, y), 3)} %')
    print(f'F1 score: {round(f1_score(y_pred, y), 3)} %')
    print(f'AUC: {round(roc_auc_score(y_pred, y), 3)} %')