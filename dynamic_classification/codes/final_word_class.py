import numpy as np
from numpy import array
import pandas as pd
import lightgbm as lgbm
import os
import sys


from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn import metrics, neighbors
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree


class Word_Class:

    # Initializer / Instance Attributes
    def __init__(self, class_name, word_represetation_algorithm, classification_algorithm, dataset, kfold):
        self.word_represetation_algorithm = word_represetation_algorithm
        self.classification_algorithm = classification_algorithm
        self.dataset = dataset
        self.kfold = kfold
        self.class_name = class_name
        dataset_class_name = dataset.groupby(self.class_name)
        self.cate = dataset_class_name.groups.keys()
        #self.cate = ["business", "entertainment", "politics", "sport", "tech"]
        self.total_df = pd.DataFrame()
        
        # Data preprocessing:
        # Importing the dataset
        self.X = self.dataset.iloc[:, 1].values
        self.y = self.dataset.iloc[:, 0].values
        self.model_names = [ "LGBM", "NB", "SVM", "DT", "KNN", "LR", "RF", "LSVC",]
        self.word_method_names = [ "tf_idf", "bag_of_words",]
        
        self.models = [  lgbm.LGBMClassifier(objective='multiclass', verbose=-1, learning_rate=0.5, max_depth=20, num_leaves=50, n_estimators=120, max_bin=2000,),
                         MultinomialNB(),
                         SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
                         tree.DecisionTreeClassifier(),
                         neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, n_jobs=1),
                         LogisticRegression(random_state=0),
                         RandomForestClassifier(n_estimators=500,verbose=1,n_jobs=-1),
                         LinearSVC(), ]
        
        self.word_methods = [ TfidfVectorizer(dtype=np.float32, sublinear_tf=True, use_idf=True, smooth_idf=True, encoding='utf-8'),
                              CountVectorizer(analyzer='word', input='content'),  ]

    def control_word_method(self):    
        for method in self.word_method_names:
            if method == self.word_represetation_algorithm: 
                if self.classification_algorithm == "LGBM" and method != "tf_idf":
                    print("LGBM sınıflandırı ile text verisi için yalnızca tf-idf kelime temsil yöntemi kullanılabilir.")
                    return 1
                return 0   
        print("Geçersiz bir kelime temsil algoritması girdiniz." +
              "\n TF-IDF = tf_idf, \n Bag of Words = bag_of_words.")
        return 1
    
    def control_classifier(self): 
        for model in self.model_names:
            if model == self.classification_algorithm: 
                return 0    
        print("Geçersiz bir sınıflandırma algoritması girdiniz. " +
              "\n Naive Bayes = NB, \n K en yakın komşu = KNN, \n Linear Regression = LR," +
              "\n Random Forest = RF, \n Decision Tree = DT, \n Linear Support Vector Machine = LSVC, " +
              "\n Support Vector Machine = SVM, \n LightGBM = LGBM.")
        return 1
    
    def fit(self):
        # if you want to add the number of fold as a parameter, you should do in here 
        # lenght = self.dataset.shape[0] # len(self.dataset.index) || len(dataset)
        # k = int(lenght / 200)
        
        
        # if the name of word representation algorithm or classification algorithm is wrong, process is terminated.
        if self.control_word_method() or self.control_classifier():
            return 
        
        self.skfold()
        
    def skfold(self):
        # prepare cross validation  
        skfold = StratifiedKFold(n_splits = self.kfold, random_state=None, shuffle=False)
        cntrl = 1
        print("Word Method: ", self.word_represetation_algorithm, " and Classifier: ",  self.classification_algorithm)
        for x_train, x_test in skfold.split(self.X, self.y):
            # fit the chosen word representation method
            X_train_wm, X_test_wm = self.fit_word_method(x_train, x_test)
            
            # fit the chosen classifier
            Y_test, predicted_Y = self.fit_classifier(x_train, x_test, X_train_wm, X_test_wm)
            
            # calculate scores
            accuracy, cm, f_measure = self.scores(Y_test, predicted_Y)

            # prepare the total cross validation table 
            if cntrl == 1:
                self.total_df = cm
            else:
                self.total_df = self.total_df.add(cm, fill_value=0)
            
            print(cntrl, ". Fold = Accuracy: ", round(accuracy, 2), ", F-measure: ", round(f_measure, 2))
            # print(cntrl, ". Fold = Confusion Matrix: ")
            # print(cm)
            
            cntrl += 1
            
    def fit_word_method(self, x_train, x_test):
        # get data
        X_train, X_test = self.X[x_train], self.X[x_test]
        
        for i in range(len(self.word_method_names)):
            if self.word_method_names[i] == self.word_represetation_algorithm:
                vec = self.word_methods[i]

        X_train_wm = vec.fit_transform(X_train)
        X_test_wm = vec.transform(X_test)
        #  print(vec.get_feature_names())
        #  print(X_train_wm.toarray())
            
        return X_train_wm, X_test_wm   
    
    def fit_classifier(self, x_train, x_test, X_train_wm, X_test_wm):      
        # get data
        Y_train, Y_test = self.y[x_train], self.y[x_test]
        
        for i in range(len(self.model_names)):
            if self.model_names[i] == self.classification_algorithm:
                clf = self.models[i].fit(X_train_wm, Y_train)
                predicted_Y = clf.predict(X_test_wm) 
        
        return Y_test, predicted_Y
    
    def scores(self, Y_test, predicted_Y):
        
        # Calculate confusion matrix
        cm = confusion_matrix(Y_test, predicted_Y)
        
        # convet to confusion matrix to confusion dataframe
        cm = self.cm_to_df(cm, self.cate)

        # Calculate accuracy score
        accuracy = accuracy_score(Y_test, predicted_Y)
        
        # Calculate f measure score
        # macro = Calculate metrics for each label, and find their unweighted mean. 
        #         This does not take label imbalance into account.
        f_measure = f1_score(Y_test, predicted_Y, average = 'macro')
    
        return accuracy, cm, f_measure
    
    # convert confusion matrix to dataframe 
    def cm_to_df(self, cm, labels):
        df = pd.DataFrame()
        # rows
        for i, row_label in enumerate(labels):
            rowdata={}
            
            # columns
            for j, col_label in enumerate(labels): 
                rowdata[col_label] = cm[i,j]
            df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient = 'index'))
        return df[labels]

"""
 Geçerli kelime temsil algoritmaları:
      TF-IDF = tf_idf,
      Bag of Words = bag_of_words.

 Geçerli siniflandirma algoritmaları:
      Naive Bayes = NB, 
      K en yakın komşu = KNN, 
      Linear Regression = LR,
      Random Forest = RF,
      Decision Tree = DT, 
      Linear Support Vector Machine = LSVC,
      Support Vector Machine = SVM, 
      LightGBM = LGBM.
"""

path = os.path.dirname(__file__)
path , file = os.path.split(path)
path = path + "/data/bbc-text.csv"


dataset = pd.read_csv(path)

# example1
wc = Word_Class("category", "bag_of_words", "DT", dataset, 5)
wc.fit()
# total confusion matrix
print(wc.total_df)

# example2
wc = Word_Class("category", "tf_idf", "NB", dataset, 4)
wc.fit()
# total confusion matrix
print(wc.total_df)
1
# example3
wc = Word_Class("category", "tf_idf", "LGBM", dataset, 3)
wc.fit()
# total confusion matrix
print(wc.total_df)

