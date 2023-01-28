import pandas as pd
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, rdmolfiles, rdmolops
from rdkit import DataStructs
from tensorflow.keras.optimizers import SGD, Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense, Activation,GaussianNoise
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score ,accuracy_score
from scipy import interp
from scipy import stats
from sklearn import linear_model
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import joblib
import optuna
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import cv
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


file_dir='D:/ROSALIND_problems/Servier_test/dataset_single.csv'
output_dir = 'D:/ROSALIND_problems/Servier_test'



def computeMorganFP(smile_string, radius=2, size=2048):
    x = np.zeros(size)
    mol = MolFromSmiles(smile_string)
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)
    morgan = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius,
                                                            nBits=size,
                                                            useChirality=True,
                                                            useBondTypes=True,
                                                            useFeatures=False
                                                            )
    DataStructs.ConvertToNumpyArray(morgan,x)

    return x


''' Take in input train data (x) and train label (y), 
    perform random Gridsearch with stratifiedkfold and output best parameters for XGBOOST '''

def tune_xgboost (train_x,train_y):

    candidate_n_estimators = [int(x) for x in np.linspace(start=10, stop=1000, num=10)]
    candidate_colsample_bytree = [0.3 , 0.6, 0.9]
    candidate_max_depth = [int(x) for x in np.linspace(start=1, stop=10, num=5)]
    candidate_learning_rate = [0.1, 0.01, 0.05]
    candidate_subsample = [0.3 , 0.6, 0.9]
    candidate_min_child_weight = [ 1 , 5 ]

    # Add weight penalize
    unique, counts = np.unique(train_y, return_counts=True)
    class_dist_dict = dict(zip(unique, counts))
    scale_pos_weight = class_dist_dict[0] * 1.0 / class_dist_dict[1]


    # create random grid
    ### TRY RANDOMIZED GRID

    RandomizedSearchCV_parameters = {'n_estimators': candidate_n_estimators,
                                     'colsample_bytree':candidate_colsample_bytree ,
                                     'max_depth':candidate_max_depth,
                                     'learning_rate' : candidate_learning_rate,
                                     'subsample' : candidate_subsample,
                                     'min_child_weight' :  candidate_min_child_weight
                                     }

    # create random grid
    ### TRY RANDOMIZED GRID

    xgb_RandomizedSearchCV = XGBClassifier( max_depth=candidate_max_depth,
                                            scale_pos_weight=scale_pos_weight,
                                            learning_rate=candidate_learning_rate,
                                            colsample_bytree=candidate_colsample_bytree,
                                            subsample=candidate_subsample,
                                            n_estimators=candidate_n_estimators,
                                            min_child_weight=candidate_min_child_weight )


    cv = StratifiedKFold(n_splits=3)
    RandomizedCV_search = RandomizedSearchCV(estimator=xgb_RandomizedSearchCV, param_distributions =RandomizedSearchCV_parameters, cv=cv, verbose=2,
                                n_jobs=-1,random_state=1,n_iter=50,scoring='roc_auc')
    # Fit the random search model
    RandomizedCV_search.fit(train_x, train_y)

    model_random_grid_search= RandomizedCV_search.best_estimator_
    model_grid_bestparams=dict(RandomizedCV_search.best_params_ ) # output best param as dict

    return model_random_grid_search,model_grid_bestparams




''' Take in input train data (x) and train label (y), 
    perform random Gridsearch with stratifiedkfold and output best parameters for Linear model '''

def tune_lr (x_train,y_train):
    # Number of trees in random forest
    candidate_C = [0.01, 0.5, 1, 10, 50, 10,100,500,1000]
    candidate_penalty =  ['elasticnet', 'l1', 'l2', 'none', 'newton-cg', 'lbfgs']
    candidate_max_iter =  [50, 100, 200, 500, 1000, 2500]
    candidate_solver= ['newton-cg', 'lbfgs','saga']

    RandomizedSearchCV_parameters = [
        {'solver': ['saga'],
         'penalty': ['elasticnet', 'l1', 'l2', 'none'],
         'max_iter': [50, 100, 200, 500, 1000, 2500],
         'C': candidate_C},
        {'solver': ['newton-cg', 'lbfgs'],
         'penalty': ['l2', 'none'],
         'max_iter': [50, 100, 200, 500, 1000, 2500],
         'C': candidate_C}
    ]

    ### TRY RANDOMIZED GRID

    lr_RandomizedSearchCV = linear_model.LogisticRegression(class_weight='balanced')


    cv = StratifiedKFold(n_splits=3)
    RandomizedCV_search = RandomizedSearchCV(lr_RandomizedSearchCV, RandomizedSearchCV_parameters,  cv=cv, verbose=2,
                                n_jobs=-1,random_state=1,n_iter=50,scoring='roc_auc')
    # Fit the random search model
    RandomizedCV_search.fit(x_train, y_train)

    model_random_grid_search = RandomizedCV_search.best_estimator_ # output model with best param
    model_grid_bestparams = dict(RandomizedCV_search.best_params_ ) # output best param as dict
    print("Best parameter (CV score=%0.3f):" % RandomizedCV_search.best_score_)

    return model_random_grid_search , model_grid_bestparams



''' Take in input train data (x) and train label (y), 
    perform random Gridsearch with stratifiedkfold and output best parameters for RandomForest model '''

def tune_rf (train_x,train_y):
    # Number of trees in random forest
    candidate_n_estimators = [int(x) for x in np.linspace(start=200, stop=1000, num=10)]
    # Number of features to consider at every split
    candidate_max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    candidate_max_depth = [int(x) for x in np.linspace(10, 100, num=10)]
    candidate_max_depth.append(None)
    # Minimum number of samples required to split a node
    candidate_min_samples_split  = [2, 5, 7, 10, 12, 14]
    # Minimum number of samples required at each leaf node
    candidate_min_samples_leaf = [1, 2, 3, 4, 5 , 6]
    # Method of selecting samples for training each tree
    candidate_bootstrap = [True, False]
    candidate_criterion = ['gini', 'entropy']
    # create random grid
    ### TRY RANDOMIZED GRID
    RandomizedSearchCV_parameters = {'n_estimators': candidate_n_estimators,
                                   'max_features': candidate_max_features,
                                   'max_depth':candidate_max_depth ,
                                   'min_samples_split':candidate_min_samples_split ,
                                   'min_samples_leaf': candidate_min_samples_leaf,
                                   'bootstrap': candidate_bootstrap,
                                   'criterion' :candidate_criterion}






    rf_RandomizedSearchCV = RandomForestClassifier(n_estimators=candidate_n_estimators,
                                                   max_depth=candidate_max_depth,
                                                   oob_score=True,
                                                    class_weight='balanced',
                                                   max_features=candidate_max_features,
                                                   criterion=candidate_criterion,
                                                   bootstrap=candidate_bootstrap,
                                                   min_samples_split=candidate_min_samples_split,
                                                   min_samples_leaf=candidate_min_samples_leaf)

    # search across 100 different combinations, and use all available cores
    cv = StratifiedKFold(n_splits=3)
    RandomizedCV_search = RandomizedSearchCV(estimator=rf_RandomizedSearchCV, param_distributions =RandomizedSearchCV_parameters,  cv=cv, verbose=2,
                                n_jobs=-1,random_state=1,n_iter=50,scoring='roc_auc')
    # Fit the random search model
    RandomizedCV_search.fit(train_x, train_y)

    model_random_grid_search= RandomizedCV_search.best_estimator_
    model_grid_bestparams=dict(RandomizedCV_search.best_params_ ) # output best param as dict



    return model_random_grid_search,model_grid_bestparams




''' found best parameters with grid search/ cross validation, 
    long run ~ 5hours , for 3 split slice , and output the best parameters with the best score '''

def run_all_thunes(data,label): # call function with data = x_train and label = y_train of initial split
    auc_scores_test = dict( rf=[],lr=[],xgb=[])
    auc_scores_train = dict( rf=[],lr=[],xgb=[])

    recall_scores_test = dict( rf=[],lr=[],xgb=[])
    recall_scores_train = dict( rf=[],lr=[],xgb=[])

    F1_scores_test=dict( rf=[],lr=[],xgb=[])
    F1_scores_train=dict( rf=[],lr=[],xgb=[])

    Precision_scores_test=dict( rf=[],lr=[],xgb=[])
    Precision_scores_train=dict( rf=[],lr=[],xgb=[])
    # precision F1 , recall, des scores issues de la matrices de confusion , au lieu de


    for rand in [126, 84, 42] :
        print( ' Running slice Rand : ', rand)
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=rand,stratify=label) # train/test

        ## Logistic Regression
        print('Logistic Regression ')
        model_lr, params_dict = tune_lr(x_train, y_train)

        # AUC
        auc_scores_test['lr'].append((roc_auc_score(y_test, model_lr.predict_proba(x_test)[:, 1]), params_dict))  # test
        auc_scores_train['lr'].append((roc_auc_score(y_train, model_lr.predict_proba(x_train)[:, 1]), params_dict))  # train
        # F1 / recall / Precision
        prediction = model_lr.predict(x_test)
        F1_scores_test['lr'].append(f1_score(y_test, prediction))  # test
        recall_scores_test['lr'].append(recall_score(y_test, prediction))
        Precision_scores_test['lr'].append(precision_score(y_test, prediction))

        prediction = model_lr.predict(x_train)
        F1_scores_train['lr'].append(f1_score(y_train, prediction))  # train
        recall_scores_train['lr'].append(recall_score(y_train, prediction))
        Precision_scores_train['lr'].append(precision_score(y_train, prediction))


        ## Random Forest :
        print('Random forest ')
        model_rf,params_dict=tune_rf(x_train,y_train)
        # AUC
        auc_scores_test['rf'].append((roc_auc_score(y_test,model_rf.predict_proba(x_test)[:, 1]),params_dict)) # test
        auc_scores_train['rf'].append((roc_auc_score(y_train,model_rf.predict_proba(x_train)[:, 1]),params_dict)) # train
        # F1 / recall / Precision
        prediction = model_rf.predict(x_test)
        F1_scores_test['rf'].append(f1_score(y_test, prediction ))                # test
        recall_scores_test['rf'].append(recall_score(y_test, prediction ))
        Precision_scores_test['rf'].append(precision_score(y_test, prediction ))

        prediction = model_rf.predict(x_train)
        F1_scores_train['rf'].append(f1_score(y_train, prediction ))                # train
        recall_scores_train['rf'].append(recall_score(y_train, prediction ))
        Precision_scores_train['rf'].append(precision_score(y_train, prediction ))



        # XGBOOST
        print (' XGBOOST ')
        model_xgb,params_dict=tune_xgboost(x_train,y_train)

        # AUC
        auc_scores_test['xgb'].append((roc_auc_score(y_test, model_xgb.predict_proba(x_test)[:, 1]),params_dict))  # test
        auc_scores_train['xgb'].append((roc_auc_score(y_train, model_xgb.predict_proba(x_train)[:, 1]),params_dict))  # train
        # F1 / recall / Precision

        prediction = model_xgb.predict(x_test)
        F1_scores_test['xgb'].append(f1_score(y_test, prediction))  # test
        recall_scores_test['xgb'].append(recall_score(y_test, prediction))
        Precision_scores_test['xgb'].append(precision_score(y_test, prediction))

        prediction = model_xgb.predict(x_train)
        F1_scores_train['xgb'].append(f1_score(y_train, prediction))  # train
        recall_scores_train['xgb'].append(recall_score(y_train, prediction))
        Precision_scores_train['xgb'].append(precision_score(y_train, prediction))


    # get the parameters of the best slice  for each model
    tuned_params = dict(rf=None,lr=None,xgb=None)
    for key in auc_scores_test.keys():
        tuned_params[key] = max(auc_scores_test[key])[1]
    print('**** END **** ')

    return tuned_params


''' Output statistical significance of AUC among all 3 model'''

def find_best_model_AUC(x_train, y_train, tuned_params):
    auc_scores = dict(lr=[], xgb=[], rf=[])
    statistical_result = dict(lr_vs_xgb=[], lr_vs_rf=[], xgb_vs_rf=[])

    for rand in [126, 84, 42, 21, 11, 6 , 3]:
        print(' Running slice Rand : ', rand)
        x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2, random_state=rand,
                                                            stratify=label)  # train/test

        # Logistic :
        logistic_clf = linear_model.LogisticRegression(penalty=tuned_params['lr']['penalty'],
                                                       C=tuned_params['lr']['C'],
                                                       solver=tuned_params['lr']['solver'],
                                                       class_weight='balanced',
                                                       max_iter=tuned_params['lr']['max_iter'])
        logistic_clf.fit(x_train, y_train)
        # store best param inside auc_score
        auc_scores['lr'].append(roc_auc_score(y_test, logistic_clf.predict_proba(x_test)[:, 1]))
        print("Logistic Regression AUC score: {}".format(auc_scores['lr'][-1]))

        # Random forest :
        rf_clf = RandomForestClassifier(n_estimators=tuned_params['rf']['n_estimators'],
                                        max_depth=tuned_params['rf']['max_depth'],
                                        oob_score=True,
                                        class_weight='balanced',
                                        criterion=tuned_params['rf']['criterion'],
                                        max_features=tuned_params['rf']['max_features'],
                                        bootstrap=tuned_params['rf']['bootstrap'],
                                        min_samples_split=tuned_params['rf']['min_samples_split'],
                                        min_samples_leaf = tuned_params['rf']['min_samples_leaf'])
        rf_clf.fit(x_train, y_train)
        auc_scores['rf'].append(roc_auc_score(y_test, rf_clf.predict_proba(x_test)[:, 1]))
        print("Random Forest AUC score : {}".format(auc_scores['rf'][-1]))

        # XGBOOST :
        # Penalize weight
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist_dict = dict(zip(unique, counts))
        scale_pos_weight = class_dist_dict[0] * 1.0 / class_dist_dict[1]

        xgb_clf = XGBClassifier(max_depth=tuned_params['xgb']['max_depth'],
                                scale_pos_weight=scale_pos_weight,
                                learning_rate=tuned_params['xgb']['learning_rate'],
                                n_estimators=tuned_params['xgb']['n_estimators'],
                                subsample=tuned_params['xgb']['subsample'],
                                colsample_bytree=tuned_params['xgb']['colsample_bytree'],
                                min_child_weight=tuned_params['xgb']['min_child_weight'])
        xgb_clf.fit(x_train, y_train)
        auc_scores['xgb'].append(roc_auc_score(y_test, xgb_clf.predict_proba(x_test)[:, 1]))
        print("XGBClassifier AUC score : {}".format(auc_scores['xgb'][-1]))

    # T-Test
    statistical_result['xgb_vs_rf'].append(stats.ttest_rel(auc_scores['xgb'], auc_scores['rf']))
    print('*** Here is the statistics XGB AGAINST RF AUC score :',statistical_result['xgb_vs_rf'])

    statistical_result['lr_vs_xgb'].append(stats.ttest_rel(auc_scores['lr'], auc_scores['xgb']))
    print('*** Here is the statistics XGB AGAINST RF AUC score :',statistical_result['lr_vs_xgb'])


    statistical_result['lr_vs_rf'].append(stats.ttest_rel(auc_scores['lr'], auc_scores['rf']))
    print('*** Here is the statistics LR AGAINST RF AUC score :', statistical_result['lr_vs_rf'])



    return statistical_result




# use optuna as minimizing/maximizing function
'''
def objective_XGB(trial):
    data_dmatrix = xgb.DMatrix(data=x_train,
                               label=y_train)

    params = {  'max_depth': trial.suggest_int('max_depth', 2, 15),
        'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05),
        'n_estimators': trial.suggest_int('n_estimators', 1000, 10000, 100),
        'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01),
        'reg_alpha': trial.suggest_int('reg_alpha', 1, 50),
        'reg_lambda': trial.suggest_int('reg_lambda', 5, 100),
        'min_child_weight': trial.suggest_int('min_child_weight', 2, 20),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.1, 1.0),
    }
   # bst = XGBClassifier(**params)

    bst.fit(x_train, y_train, eval_set=[(x_test, y_test)],verbose=True,eval_metric=["auc"])
    auc = roc_auc_score(y_test, bst.predict_proba(x_test)[:, 1])

    xgb_cv = cv(dtrain=data_dmatrix, params=params, nfold=3, num_boost_round=2000,
                early_stopping_rounds=10, metrics="AUC", as_pandas=True, seed=123)

    return auc
#eval_metric=["auc"]
# maximize AUC score
study = optuna.create_study(direction='maximize')
study.optimize(objective_XGB, n_trials=20)
'''




''' Output metrics (  AUC; F1 score etc .. ) 
    for each model for original training and Testing set '''

def fit_and_run (x_train, y_train, y_test,x_test,final_thune,model):

    if model == 'RandomForestClassifier':
        # test thune with training set
        rf_model = RandomForestClassifier(n_estimators=final_thune['rf']['n_estimators'],
                                        max_depth=final_thune['rf']['max_depth'],
                                        oob_score=True,
                                        class_weight='balanced',
                                        criterion=final_thune['rf']['criterion'],
                                        max_features=final_thune['rf']['max_features'],
                                        bootstrap=final_thune['rf']['bootstrap'],
                                        min_samples_split=final_thune['rf']['min_samples_split'],
                                        min_samples_leaf=final_thune['rf']['min_samples_leaf'])
        rf_model.fit(x_train, y_train)
        print('*** Here is the AUC score for TRAINING',roc_auc_score(y_train, rf_model.predict_proba(x_train)[:, 1]))  # test
        print('*** Here is the AUC score for TESTING', roc_auc_score(y_test, rf_model.predict_proba(x_test)[:, 1]))  # train

        # print(confusion_matrix(y_train,rf_clf.predict(x_train) ))  # train
        # print(confusion_matrix(y_test,rf_clf.predict(x_test) ))    # test

        print( '*** Here is the classification report for TRAINING', classification_report(y_train, rf_model.predict(x_train))) # train
        print('*** Here is the classification report for TESTING',classification_report(y_test, rf_model.predict(x_test)))  # test

        # Ability for the user to predict the property P1f for any given smile
        # x_test need to be dataframe and of lenght 2048
        return rf_model


    elif model == 'linear_model':
        # test thune with training set
        lr_model = linear_model.LogisticRegression(class_weight='balanced',
                                                   solver=final_thune['lr']['solver'],
                                                   penalty=final_thune['lr']['penalty'],
                                                   max_iter=final_thune['lr']['max_iter'],
                                                    C = final_thune['lr']['C'] )
        lr_model.fit(x_train, y_train)
        print('*** Here is the AUC score for TRAINING',roc_auc_score(y_train, lr_model.predict_proba(x_train)[:, 1]))  # test
        print('*** Here is the AUC score for TESTING',roc_auc_score(y_test, lr_model.predict_proba(x_test)[:, 1]))  # train

        # print(confusion_matrix(y_train,rf_clf.predict(x_train) ))  # train
        # print(confusion_matrix(y_test,rf_clf.predict(x_test) ))    # test

        print('*** Here is the classification report for TRAINING',classification_report(y_train, lr_model.predict(x_train)))  # train
        print('*** Here is the classification report for TESTING', classification_report(y_test, lr_model.predict(x_test)))  # test

        return lr_model

    elif model == 'XGBClassifier' :
        unique, counts = np.unique(y_train, return_counts=True)
        class_dist_dict = dict(zip(unique, counts))
        scale_pos_weight = class_dist_dict[0] * 1.0 / class_dist_dict[1]

        xgb_model = XGBClassifier(max_depth=final_thune['xgb']['max_depth'],
                                               scale_pos_weight=scale_pos_weight,
                                               learning_rate=final_thune['xgb']['learning_rate'],
                                               colsample_bytree=final_thune['xgb']['colsample_bytree'],
                                               subsample=final_thune['xgb']['subsample'],
                                               n_estimators=final_thune['xgb']['n_estimators'],
                                               min_child_weight=final_thune['xgb']['min_child_weight'])
        xgb_model.fit(x_train, y_train)
        print('*** Here is the AUC score for TRAINING',roc_auc_score(y_train, xgb_model.predict_proba(x_train)[:, 1]))  # test
        print('*** Here is the AUC score for TESTING',roc_auc_score(y_test, xgb_model.predict_proba(x_test)[:, 1]))  # train

        # print(confusion_matrix(y_train,rf_clf.predict(x_train) ))  # train
        # print(confusion_matrix(y_test,rf_clf.predict(x_test) ))    # test

        print('*** Here is the classification report for TRAINING',classification_report(y_train, xgb_model.predict(x_train)))  # train
        print('*** Here is the classification report for TESTING',classification_report(y_test, xgb_model.predict(x_test)))  # test

        return xgb_model





''' Plot learning curve for AUC score based on different size of training set 
    for both testing ( validation ) and training '''

def plot_learning_curves(x_train, y_train, final_thune,direct):
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist_dict = dict(zip(unique, counts))
    scale_pos_weight = class_dist_dict[0] * 1.0 / class_dist_dict[1]
    Training_ratio = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]

    sub_train_x, sub_test_x, sub_train_y, sub_test_y = train_test_split(x_train,
                                                                        y_train,
                                                                        test_size=0.2,
                                                                        stratify=y_train)


    title_dict = {'lr': 'Logistic Regression', 'xgb': 'Gradient Boosted Decision Tree', 'rf': 'Random Forest'}
    for key in final_thune:
        auc_scores_training, auc_scores_validation = [], []

        if key == 'lr':
            print (' *** plotting lr *** ')
            model = linear_model.LogisticRegression(class_weight='balanced',
                                                       solver = final_thune['lr']['solver'],
                                                       penalty = final_thune['lr']['penalty'],
                                                       max_iter = final_thune['lr']['max_iter'],
                                                       C = final_thune['lr']['C'])

        elif key == 'xgb':
            print (' *** plotting xgb *** ')
            model = XGBClassifier(max_depth=final_thune['xgb']['max_depth'],
                                           scale_pos_weight=scale_pos_weight,
                                           learning_rate=final_thune['xgb']['learning_rate'],
                                           colsample_bytree=final_thune['xgb']['colsample_bytree'],
                                           subsample=final_thune['xgb']['subsample'],
                                           n_estimators=final_thune['xgb']['n_estimators'],
                                           min_child_weight=final_thune['xgb']['min_child_weight'])

        elif key == 'rf':
            print (' *** plotting rf *** ')
            model =  RandomForestClassifier(n_estimators=final_thune['rf']['n_estimators'],
                                    max_depth=final_thune['rf']['max_depth'],
                                    oob_score=True,
                                    class_weight='balanced',
                                    criterion=final_thune['rf']['criterion'],
                                    max_features=final_thune['rf']['max_features'],
                                    bootstrap=final_thune['rf']['bootstrap'],
                                    min_samples_split=final_thune['rf']['min_samples_split'],
                                    min_samples_leaf=final_thune['rf']['min_samples_leaf'])

        # perform subset of training set based on training_size_ratio
        for tsr in Training_ratio:
            sub_sub_train_x = sub_train_x[:int(tsr * len(sub_train_x))]
            sub_sub_train_y = sub_train_y[:int(tsr * len(sub_train_y))]
            model.fit(sub_sub_train_x, sub_sub_train_y)
            auc_scores_training.append(roc_auc_score(sub_sub_train_y, model.predict_proba(sub_sub_train_x)[:, 1]))
            auc_scores_validation.append(roc_auc_score(sub_test_y, model.predict_proba(sub_test_x)[:, 1]))

        fig = plt.figure()
        plt.plot(Training_ratio, auc_scores_validation, 'ro-', label='Testing AUC')
        plt.plot(Training_ratio, auc_scores_training, 'bo-', label='Training AUC')
        plt.legend()
        plt.ylim(0)
        plt.xticks(Training_ratio)
        plt.xlabel('Training size')
        plt.ylabel('AUC score')
        plt.title(title_dict[key])
        plt.show()
        plt.savefig(direct+'/'+key+'.png')
        plt.close(fig)





''' Plot ROC AUC score for each model '''

def plot_roc_curves(x_train, y_train, final_thune,direct):
    title_dict = {'lr': 'Logistic Regression', 'xgb': 'Gradient Boosted Decision Tree', 'rf': 'Random Forest'}
    for key in final_thune:
        if key == 'lr':
            print (' *** plotting lr *** ')

            model = linear_model.LogisticRegression(class_weight='balanced',
                                                           solver = final_thune['lr']['solver'],
                                                           penalty = final_thune['lr']['penalty'],
                                                           max_iter = final_thune['lr']['max_iter'],
                                                           C = final_thune['lr']['C'])
        elif key == 'xgb':
            print (' *** plotting xgb *** ')

            unique, counts = np.unique(y_train, return_counts=True)
            class_dist_dict = dict(zip(unique, counts))
            scale_pos_weight = class_dist_dict[0] * 1.0 / class_dist_dict[1]

            model = XGBClassifier(max_depth=final_thune['xgb']['max_depth'],
                                               scale_pos_weight=scale_pos_weight,
                                               learning_rate=final_thune['xgb']['learning_rate'],
                                               colsample_bytree=final_thune['xgb']['colsample_bytree'],
                                               subsample=final_thune['xgb']['subsample'],
                                               n_estimators=final_thune['xgb']['n_estimators'],
                                               min_child_weight=final_thune['xgb']['min_child_weight'])
        elif key == 'rf':
            print (' *** plotting rf *** ')

            model =  RandomForestClassifier(n_estimators=final_thune['rf']['n_estimators'],
                                        max_depth=final_thune['rf']['max_depth'],
                                        oob_score=True,
                                        class_weight='balanced',
                                        criterion=final_thune['rf']['criterion'],
                                        max_features=final_thune['rf']['max_features'],
                                        bootstrap=final_thune['rf']['bootstrap'],
                                        min_samples_split=final_thune['rf']['min_samples_split'],
                                        min_samples_leaf=final_thune['rf']['min_samples_leaf'])
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        i = 0
        kf = StratifiedKFold(n_splits=10)
        for fold_num, (train_x, test_y) in enumerate(kf.split(x_train, y_train)):
            print(fold_num)
            train_fold_x, train_fold_y = x_train.iloc[train_x], y_train.iloc[train_x]
            test_fold_x, test_fold_y = x_train.iloc[test_y], y_train.iloc[test_y]
            probas_ = model.fit(train_fold_x, train_fold_y).predict_proba(test_fold_x)[:, 1]

            # COMPUTE ROC AND AUC
            fpr, tpr, thresholds = roc_curve(test_fold_y, probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

            i += 1

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Random Classifier(Base)', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Mean ROC curve with variability ' + title_dict[key])
        plt.legend(loc="lower right")
        plt.show()
        plt.savefig(direct + '/' +'ROC_' + key + '.png')
        plt.close()


'''
# quick data check

# check for missing values using the isnull() method
df.isnull().sum()  # no null data
# df.dropna()
# test if fduplicate
df['smiles'].nunique() - df['smiles'].count() == 0

# plot distribution  of predictor P1
import seaborn as sns
Y = df[['P1']]
print('distribution of data is' , Y["P1"].value_counts())
# ploting
counts = Y['P1'].value_counts().rename_axis('P1_proprety').reset_index(name='count')
ax = sns.barplot(x='P1_proprety', y='count', data=counts)
# since unbalanced class, we will use oversampling

'''


def main():
    df = pd.read_csv(file_dir, sep=',')
    df['Morgan_fingerprint'] = df['smiles'].map(computeMorganFP)  # get morganFingerprint as binary array
    data = df.Morgan_fingerprint.apply(pd.Series)  # get df columns for each bit morgan
    label = df['P1']  # get label
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=42,
                                                        stratify=label)  # train/test
    #final_thune=run_all_thunes(x_train,y_train) # train the model and output dict of best param
    final_thune={'rf': {'n_estimators': 288,     # here  is the output of the best model
            'min_samples_split': 5,
            'min_samples_leaf': 6,
            'max_features': 'auto',
            'max_depth': 20,
            'criterion': 'gini',
            'bootstrap': True},
     'lr': {'solver': 'newton-cg', 'penalty': 'l2', 'max_iter': 2500, 'C': 0.01},
     'xgb': {'subsample': 0.6,
             'n_estimators': 670,
             'min_child_weight': 1,
             'max_depth': 5,
             'learning_rate': 0.01,
             'colsample_bytree': 0.3}}
    # test thune with original training and testing set
    # model available : 'linear_model' ;  'RandomForestClassifier ' ; 'XGBClassifier'

    # run with  linear model
    lr_fit = fit_and_run(x_train, y_train, y_test,x_test,final_thune,'linear_model')

    # run with  RandomForestClassifier
    rf_fit=fit_and_run(x_train, y_train, y_test,x_test,final_thune,'RandomForestClassifier')

    # run with  XGBClassifier
    xgb_fit=fit_and_run(x_train, y_train, y_test,x_test,final_thune,'XGBClassifier')

     # test statistics
    find_best_model_AUC(x_train, y_train, final_thune)

    # Plot Validation training size AUC score , last argument is directory
    plot_learning_curves(x_train, y_train, final_thune,output_dir)

    # Plot mean roc curves
    plot_roc_curves(x_train, y_train, final_thune,output_dir)


# Ability for the user to predict the property P1f for any given smile
    # x_test need to be dataframe of binary value  and of length 2048 , otherwise don't work

    xgb_predict = (xgb_fit.predict(x_test)) # XGBOOST
    print(xgb_predict)

    rf_predict = (rf_fit.predict(x_test)) # Random forest
    print(rf_predict)

    lr_predict = (lr_fit.predict(x_test)) # Linear
    print(lr_predict)



main()