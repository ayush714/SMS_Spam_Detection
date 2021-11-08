from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier


class ModelTraining:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def logistic_regression_model(self):
        print("Training the model logistic regression model:- ")
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(self.x_train, self.y_train)
        
        print("Completed training the model")
        return log_reg

    def Xgboost_model(self, fine_tuning=True):
        if fine_tuning:
            print("Started Finetuning the model:- ")
            n_estimators = [50, 100, 150, 200]
            max_depth = [2, 4, 6, 8]
            learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
            subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            min_child_weight = [1, 2, 3, 4, 5]
            gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
            }
            xgb = XGBClassifier()
            clf = RandomizedSearchCV(xgb, params, cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_train)
            print("Finished Hyperparameter search")
            return clf
        else:
            print("Started training the model:- ")
            xgb = XGBClassifier(
                learning_rate=0.3,
                max_delta_step=0,
                max_depth=8,
                min_child_weight=1,
                n_estimators=50,
                n_jobs=16,
                num_parallel_tree=1,
                random_state=0,
                reg_alpha=0,
                reg_lambda=1,
                scale_pos_weight=1,
                subsample=1.0,
                tree_method="exact",
                validate_parameters=1,
                verbosity=None,
            )
            xgb.fit(self.x_train, self.y_train)
            print("Completed training the model")
            return xgb


    def svm_model(self, fine_tuning=True): 
        if fine_tuning:  
            print("Started Finetuning the model:- ")
            C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            params = {
                "C": C,
                "gamma": gamma,
            }
            svm = SVC()
            clf = RandomizedSearchCV(svm, params, cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_train)
            print("Finished Hyperparameter search")
            return clf
        else:
            print("Started training the model:- ")
            svm = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)
            svm.fit(self.x_train, self.y_train)
            print("Completed training the model")
            return svm 

    def random_forest_model(self, finetuning=True):
        if finetuning: 
            print("Started Finetuning the model:- ")
            n_estimators = [50, 100, 150, 200]
            max_depth = [2, 4, 6, 8]
            learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
            subsample = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            colsample_bytree = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
            min_child_weight = [1, 2, 3, 4, 5]
            gamma = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            params = {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "min_child_weight": min_child_weight,
                "gamma": gamma,
            }
            rf = RandomForestClassifier()
            clf = RandomizedSearchCV(rf, params, cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_train)
            print("Finished Hyperparameter search")
            return clf

    def Naive_Bayes(self):
        print("Training the model Naive Bayes:- ")
        nb_model = MultinomialNB()
        nb_model.fit(self.x_train, self.y_train)
        print("Completed training the model")
        return nb_model

    def KNN_model(self, finetuning = True): 
        if finetuning: 
            print("Started Finetuning the model:- ")
            n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            weights = ['uniform', 'distance']
            algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
            leaf_size = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            p = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            params = {
                "n_neighbors": n_neighbors,
                "weights": weights,
                "algorithm": algorithm,
                "leaf_size": leaf_size,
                "p": p,
            }
            knn = KNeighborsClassifier()
            clf = RandomizedSearchCV(knn, params, cv=5, n_jobs=-1)
            clf.fit(self.x_train, self.y_train)
            print("Finished Hyperparameter search")
            return clf
        else:
            print("Started training the model:- ")
            knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)
            knn.fit(self.x_train, self.y_train)
            print("Completed training the model")
            return knn



