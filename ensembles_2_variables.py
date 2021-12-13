from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def ensembles_2_variables(type, parameters, X, y):
    
    '''
    This function allows to the user to find the best parameters for an ensambler classifier (Randomforest y XGBoost)
    of 2 classes, easily
    Parameters:
    - type: Type of model. 'rfc' for RandomForestClassifier or 'xgbc' for XGBClassifier
    - parameters: for the iteration. They will be these two of the following example:
            parameters = {
            'n_estimators': [100,150,200,250,300,400,500,1000], 
            'max_depth': [5,10,15,20,25,30,35,40]
            }
        They must be in a dictionary like it is shown.

    - X, y: Values for the model. The function divides them into train and test.
    The function doesn't return anything, just shows the optimum values for the model in a table and in some plots with the help
    of the function 'ensemble_tester'.
    '''
    
    rf_scores = pd.DataFrame(columns=('n_estimators','max_depth','train_auc','test_auc'))
    cols = list(rf_scores)
    rf_data = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    for n in parameters['n_estimators']:
        for depth in parameters['max_depth']:
            if type == 'rfc':
                aucs = ensemble_tester(X_train, y_train, X_test, y_test, cl = 'rf', balanced = True, n_estimators=n, max_depth=depth)
            elif type == 'xgbc':
                aucs = ensemble_tester(X_train, y_train, X_test, y_test, cl = 'xg', balanced = True, n_estimators=n, max_depth=depth)
            
            values = [n, depth, aucs[0], aucs[1]]
            zipped = zip(cols, values)
            param_dict = dict(zipped)
            rf_data.append(param_dict)
    rf_scores = rf_scores.append(rf_data, True)

    print(rf_scores[rf_scores['train_auc']==rf_scores['train_auc'].max()])
    print(rf_scores[rf_scores['test_auc']==rf_scores['test_auc'].max()])

    fig, ax = plt.subplots(1,2, figsize=(10,5), sharey=True)
    sns.lineplot(data=rf_scores, x='n_estimators', y='train_auc', ax=ax[0], label='Train AUC')
    sns.lineplot(data=rf_scores, x='n_estimators', y='test_auc', ax=ax[0], label='Test AUC')
    ax[0].set_ylabel='AUC'
    ax[0].legend()

    sns.lineplot(data=rf_scores, x='max_depth', y='train_auc', ax=ax[1], label='Train AUC')
    sns.lineplot(data=rf_scores, x='max_depth', y='test_auc', ax=ax[1], label='Test AUC')
    ax[1].legend()
    plt.show()

def ensemble_tester(X_train, y_train, X_test, y_test, cl, balanced = True, **params):
    '''This function receives the values of X and y to create a classifier model and returns the AUC score.
    Parameters:
    - X_train: values for training
    - y_train: labels for training
    - X_test: values for testing
    - y_test: labels for testing
    - cl: class of model: 'rfc' or 'xgbc'
    - balanced: True if the label is balanced or False if it is imbalanced. True by default.
    - **params
    '''
    if cl == 'rfc':
        if balanced == True:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, **params)
            model.fit(X_train, y_train)
        else:
            model = RandomForestClassifier(random_state=0, n_jobs=-1, class_weight="balanced", **params)
            model.fit(X_train, y_train)
    
    elif cl == 'xg':
        if balanced == True:
            rf = xgboost.XGBClassifier(random_state=0, n_jobs=-1, **params)
            rf.fit(X_train, y_train)
        else:
            rf = xgboost.XGBClassifier(random_state=0, n_jobs=-1, scale_pos_weight=(y_train.value_counts()[0]) / y_train.value_counts()[1], **params)
            rf.fit(X_train, y_train)
    accs = roc_auc_score(y_train, model.predict(X_train)), roc_auc_score(y_test, model.predict(X_test))
    return accs