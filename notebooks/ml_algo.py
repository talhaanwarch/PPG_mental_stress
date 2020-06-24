from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,scale,MaxAbsScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy as np
def knn_param_selection(X, y,cv=5):
    n_neighbors  = [ 3,4,5,6,7,8,9]
    weights  = ['uniform','distance']
    metric=['minkowski','manhattan','euclidean']
    param_grid = {'clf__n_neighbors': n_neighbors, 'clf__weights' : weights,'clf__metric':metric}
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
    grid_search =GridSearchCV(pipe , param_grid,  cv=cv,n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_params_)
    return grid_search.best_score_
    
def svc_param_selection(X, y,cv=5):
    Cs = [ 0.1, 1,2,3,4,5, 10,15,20,25,30,50,70,100]
    gammas = [0.001, 0.01, 0.1,0.3,0.5, 1]
    param_grid = {'clf__C': Cs, 'clf__gamma' : gammas}
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', SVC())])
    grid_search =GridSearchCV(pipe , param_grid,  cv=cv,n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    return grid_search.best_params_
    
def logistic_param_selection(X, y, cv=5):
    C= [0.1, 1,3,5,8, 10,12,15]
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))])
    param_grid = {'clf__C': C}
    grid_search = GridSearchCV(pipe, param_grid, cv=cv,n_jobs=-1)
    #print(grid_search.get_params().keys())
    grid_search.fit(X, y)
    
    print(grid_search.best_score_)
    return grid_search.best_params_
    
from sklearn.tree import DecisionTreeClassifier
def dtree_param_selection(X,y,cv=5):
    #create a dictionary of all values we want to test
    param_grid = { 'clf__criterion':['gini','entropy'],'clf__max_features':["auto", "sqrt", "log2"],'clf__max_depth': np.arange(2, 20),'clf__random_state':[10,20,30,40,50]}
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', DecisionTreeClassifier())])
    grid_search =GridSearchCV(pipe , param_grid,  cv=cv,n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    return grid_search.best_params_
    
from sklearn.ensemble import RandomForestClassifier
def rf_param_selection(X, y,cv=5):
    n_estimators = [50,100,150 ]
    max_depth = [5, 8, 15,]
    min_samples_split = [2, 5, 10, ]
    min_samples_leaf = [1, 2, 5] 
    max_features = ['auto', 'sqrt']
    random_state=[10,20,30,40,50]
    bootstrap = [True, False]
    param_grid = dict(clf__n_estimators = n_estimators, clf__max_depth = max_depth,clf__max_features=max_features,  
              clf__min_samples_split = min_samples_split, clf__bootstrap=bootstrap,
             clf__min_samples_leaf = min_samples_leaf,clf__random_state=random_state)
    
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier())])
    grid_search =GridSearchCV(pipe , param_grid,  cv=cv,n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_score_)
    return grid_search.best_params_