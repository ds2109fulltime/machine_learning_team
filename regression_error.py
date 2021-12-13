from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import numpy as np

def regression_errors(y, y_pred):
    '''
    This function receives the parameters 'y', 'y_pred' and prints the MAE, MSE, RMSE and R2 score.
    Parameters:
    - y: y values for the test
    - y_pred: predicted values from X values for the test
    '''

    print('MAE:', mean_absolute_error(y, y_pred))
    print("MSE:", mean_squared_error(y, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
    print('R2 score', r2_score(y, y_pred))