def classify(features_train, labels_train):   
    ### import the sklearn module for GaussianNB
    ### create classifier
    ### fit the classifier on the training features and labels
    ### return the fit classifier

        
    ### your code goes here!
    
        
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', C = 100, gamma = 100.0)
    return clf.fit(features_train, labels_train)
    
