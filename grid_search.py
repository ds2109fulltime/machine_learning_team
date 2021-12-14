def grid_search(X_train, y_train):

    '''This function performs a grid search with RandomForestClassifier, LogisticRegression and SVC
    to find the best parameters after making train_test_split.'''

    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV

    pipe = Pipeline(steps=[
        ('classifier', RandomForestClassifier())
    ])

    logistic_params = {
        'classifier': [LogisticRegression()],
        'classifier__penalty': ['l1', 'l2']
    }

    random_forest_params = {
        'classifier': [RandomForestClassifier()],
        'classifier__max_features': [1,2,3]
    }

    svm_param = {
        'classifier': [SVC()],
        'classifier__C': [0.001, 0.1, 0.5, 1, 5, 10, 100],
    }

    search_space = [
        logistic_params,
        random_forest_params,
        svm_param
    ]

    clf = GridSearchCV(estimator = pipe,
                    param_grid = search_space,
                    cv = 10)

    clf.fit(X_train, y_train)
    print(clf.best_estimator_)
    print(clf.best_score_)
    print(clf.best_params_)
