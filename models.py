def models(X_train, X_test, y_train, y_test):
    '''Esta función compara e imprime resultados de varios modelos de ML tras hacer train_test_split()'''

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score
    from sklearn.model_selection import cross_val_score

    models = []
    models.append(['XGBClassifier',XGBClassifier(learning_rate=0.1,objective='binary:logistic',random_state=0,eval_metric='mlogloss')])
    models.append(['Logistic Regression',LogisticRegression(random_state=0)])
    models.append(['SVC',SVC(random_state=0)])
    models.append(['KNeigbors',KNeighborsClassifier()])
    models.append(['GaussianNB',GaussianNB()])
    models.append(['DecisionTree',DecisionTreeClassifier(random_state=0)])
    models.append(['RandomForest',RandomForestClassifier(random_state=0)])
    models.append(['AdaBoostClassifier',AdaBoostClassifier()])
    lst_1 = []
    for m in range(len(models)):
        lst_2 = []
        model = models[m][1]
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test,y_pred)
        accuracies = cross_val_score(estimator= model, X = X_train, y = y_train, cv=10)

    # k-fOLD Validation
        roc = roc_auc_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1 = f1_score(y_test,y_pred)
        print(models[m][0],':')
        print(cm)
        print('Accuracy Score: ',accuracy_score(y_test,y_pred))
        print('')
        print('K-Fold Validation Mean Accuracy: {:.2f} %'.format(accuracies.mean()*100))
        print('')
        print('Standard Deviation: {:.2f} %'.format(accuracies.std()*100))
        print('')
        print('ROC AUC Score: {:.2f} %'.format(roc))
        print('')
        print('Precision: {:.2f} %'.format(precision))
        print('')
        print('Recall: {:.2f} %'.format(recall))
        print('')
        print('F1 Score: {:.2f} %'.format(f1))
        print('-'*40)
        print('')
        lst_2.append(models[m][0])
        lst_2.append(accuracy_score(y_test,y_pred)*100)
        lst_2.append(accuracies.mean()*100)
        lst_2.append(accuracies.std()*100)
        lst_2.append(roc)
        lst_2.append(precision)
        lst_2.append(recall)
        lst_2.append(f1)
        lst_1.append(lst_2)