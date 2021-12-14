
def basic_regression_models(model, X_train, y_train, degree = None, alpha = None, l1_ratio = None):
    
    '''This function receives the type of model and 'X','y' train values to train the basic regression models.
    Parameters:
     - model: the type of model the user wants to train. Values: LinearRegression, PolynomialRegression, Ridge, Lasso, ElasticNet.
     - X_train: X values for training. It has to be an iterable.
     - y_train: yvalues for training. It has to be an iterable.
     - degree: just in case of model = PolynomialRegression. It must be an Integer.
     - alpha: just in case of model = Ridge, Lasso or ElasticNet.
     - l1_ratio: just in case of model = Elasticnet.
     '''
    
    if model == 'LinearRegression':
        lm = LinearRegression()
        trained_model = lm.fit(X_train, y_train)
    
    elif model == 'PolynomialRegression':
        # Preprocessing
        poly_feats = PolynomialFeatures(degree = degree)
        poly_feats.fit(X_train)
        X_poly = poly_feats.transform(X_train)

        # Train
        pol_reg = LinearRegression()
        trained_model = pol_reg.fit(X_poly, y_train)
    
    elif model == 'Ridge':
        ridge = Ridge(alpha = alpha)
        trained_model = ridge.fit(X_train, y_train)
    
    elif model == 'Lasso':
        lasso = Lasso(alpha = alpha)
        trained_model = lasso.fit(X_train, y_train)
    
    elif model == 'ElasticNet':
        elastic = ElasticNet(alpha = alpha, l1_ratio = l1_ratio)
        trained_model = elastic.fit(X_train, y_train)
    
    else:
        print('Error. Chose one of these models: LinearRegression or PolynomialRegression')
    
    return trained_model
