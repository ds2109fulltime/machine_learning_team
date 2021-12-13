# DATA PREPROCESSING FUNCTIONS



# Function to get categorical and numerical data from a given DataFrame
def columns_dtype(data, include='all', cardinality_threshold=None, return_df=False):
    """
    Returns categorical, numerical or both kind of variables based on their 
    dtype, list-like or DataFrame, based on the user specified parameter.

    Params:
        - data: Pandas DataFrame
            Pandas DataFrame where we want to analyze the variables.
        - include: str, default 'all'
            String that indicates which dtype the user wants to return.
            'all' returns both numerical and categorical variables.
            'categorical' returns only categorical variables.
            'numerical' returns only numerical variables.
        - cardinality_threshold: int, default None
            Number of maximum unique values that the categorical variable 
            has to be. The variables with higher cardinality that the 
            specified by this parameter will be ignored.
        - return_df: bool, default False
            Whether the return will be a DataFrame or not. If True, returns 
            the Pandas DataFrame with only the specified variables. If False, 
            returns a list for each of the specified variables.


    Returns:
        - List of variables or DataFrame, depending of the return_df parameter.

    """

    # Capturing the variables depending of the user's selection

    # Include All
    if include == 'all':
        if cardinality_threshold:
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and data[col].nunique() <= cardinality_threshold]
        else:
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object']

        numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]
        data_columns = categorical_cols + numerical_cols
        
        if return_df:
            return data[data_columns]

        return categorical_cols, numerical_cols

    # Include categorical only
    elif include == 'categorical':
        if cardinality_threshold:
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object' and data[col].nunique() <= cardinality_threshold]
        else:
            categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
        
        if return_df:
            return data[categorical_cols]

        return categorical_cols

    # Include numerical only        
    elif include == 'numerical':
        numerical_cols = [col for col in data.columns if data[col].dtype in ['float64', 'int64']]

        if return_df:
            return data[numerical_cols]

        return numerical_cols



# Function to scale and splitting the data to train ML models
def split_and_scale(data, target_col_name=None, scaling_method='standard', time_series=False, test_size=0.2):
    """
    Split the data into train and validation data. If the data 
    will train a model for time series, the split will not have 
    shuffling. Once the data is splitted, scale the data (only if 
    the data is not time series data). The user must choose by 
    parameter the scaling method (standard or min-max scaling).

    Params:
        - data: expected Pandas DataFrame.
        - target_col_name: str, default None
            name of the target variable.
        -  scaling_method: str, default 'standard'
            allowed 'minmax' or 'standard'.
        - time_series: bool, default False
            if True, shuffle and scaling will be False.
        - test_size: float, default 0.2
            percentage size of the validation data (0 to 1).

    Returns:
        - Splitted —and scaled, if selected— data. 
    
    """

    from sklearn.model_selection import train_test_split

    # Getting Features and Target
    X = data.drop(target_col_name, axis=1)
    y = data[target_col_name]

    if time_series:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False)
        
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=True)

        # Scaling the data
        if scaling_method == 'standard':
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            # Scaling the data (Standard Scaler)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        elif scaling_method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler

            scaler = MinMaxScaler()

            # Scaling the data (MinMax Scaler)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test



# Function to encode categorical variables
def categorical_encoding(data, kind='label_encoding'):
    """
    Encodes categorical data from a given DataFrame with one of the 
    following methods: Label Encoding or One Hot Encoding. This function 
    makes use of the scikit-learn library.

    Params:
        - data: expected Pandas DataFrame
        - kind: str, default 'label_encoding'
            one of both 'label_encoding' or 'one_hot_encoding' must be passed

    Returns:
        - Pandas DataFrame with categorical data encoded

    """

    # Checking if kind variable is correct:
    if kind not in ['label_encoding', 'one_hot_encoding']:
        raise ValueError ('"kind" parameter must be "label_encoding" or "one_hot_encoding"')

    # Checking the encoding
    if kind == 'label_encoding':
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
        for column in categorical_columns:
            data[column] = le.fit_transform(data[column])
        
        # Returning label-encoded data
        return data

    elif kind == 'one_hot_encoding':
        from sklearn.preprocessing import OneHotEncoder
        import pandas as pd

        # Getting categorical columns
        categorical_columns = [col for col in data.columns if data[col].dtype == 'object']

        oh = OneHotEncoder(handle_unknown='ignore', sparse=False)

        oh_encoded_dataframe = pd.DataFrame(oh.fit_transform(data[categorical_columns]))
        # Putting back the indexes
        oh_encoded_dataframe.index = data.index
        # Removing categorical columns to replace with onehot-encoded columns
        numeric_data = data.drop(categorical_columns, axis=1)
        # Adding categorical columns to numerical features
        onehot_encoded_data = pd.concat([numeric_data, oh_encoded_dataframe], axis=1)

        # Returning onehot-encoded data
        return onehot_encoded_data