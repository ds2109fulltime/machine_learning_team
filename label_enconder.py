def label_encoder(data,column):
    
    '''
    This functions is a easy and quickly mode to apply the label enconder to transform columns of your data. 
    
    data: dataframe
    column: column to be represented.
    '''

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(data[column])
    data[column]=le.transform(data[column])
    return data[column]