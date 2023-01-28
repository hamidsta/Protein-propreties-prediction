# Servier Data Science Technical Test
<sub>Sta Abdelhamid M2 Genomics Informatics and Mathematics for Health and Environment,  Paris-Saclay university </sub>

---


## Objective

Develop a Machine Learning model to predict P1 properties (0 , 1 )  of a molecule given Morgan_fingerprint as input ( Binary Vector ) 

Since it is a binary classification problem , i decide  to implement 3 model , from the most basic to the more "complex". The model are :

* linear_model.LogisticRegression
 
* RandomForestClassifier 
 
* XGBClassifier  

---
## Running the model : 
inside main.py function, set:
* _file_dir_ as path of csv with data to process 
* _output_dir_  as folder path to store plot

```python
file_dir='D:/ROSALIND_problems/Servier_test/dataset_single.csv'
output_dir = 'D:/ROSALIND_problems/Servier_test'
```


---

### Data Preprocessing 

* Check NA 

* Check Duplicate value

* Transform Smile molecule representation into binary vector representing the molecule structure (MorganFingerPrint)

* Check label balance 

* No normalization/scaling have been realised since the value are binary 

---

### hyperparameters tuning
In order to get the best hyperparameters, **Randomgridsearch** have been used , since it has been highlighted as a powerful way to tune the hyperparameters, and represent an excellent trade off between the computation time and the 'quality' of the hyperparameters. The iteration number have been set to *n_iter=50* .  The cross validation have been realized through *StratifiedKFold* with *n_splits=3* has a way to stratify the sampling by the class label since the label are unbalance. All 3 model have been tuned : 

* **tune_xgboost()** : That function will output XGBOOST model with best parameters by performing randomgridsearch with stratified 3-fold. I also put *scale_pos_weight* parameters to my ratio of label since it's unbalance . 

* **tune_lr()** : That function will output linear model with the best parameters. *Weight* parameters  have been set to 'balance'. 

* **tune_rf()** :  That function will output RandomForest model with the best parameters. *Weight* parameters have been set to 'balance'. 

All of those 3 tune function will be set inside :

* **run_all_thunes()** : This function will run the 3 tune function on 3 different slice of rand [126, 84, 42] . By relying on only one slice, i take the risk to run my model on an 'advantageous' slice, and it constitute a biais in my opinion. The output will be the best set of hyperparameters for each 3 model who maximise the AUC score. The Output will be store and re used for computing the other function, as depicted below : 

```python

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
```
---

### Train and evaluate the model 

* **fit_and_run()**  : Will take as input the best Hyper parameters for each model.Inside this function,  the best parameters for each 3 model of the function 'run_all_thunes' will be run in order to test the AUC score and classification report  for the initial training set and the initial testing set , in order to check if it overfit or not . Here is the related output : 

* For **Linear regression** :   _lr_fit = fit_and_run(x_train, y_train, y_test,x_test,final_thune,'linear_model')_

	* Here is the AUC score for TRAINING : 0.8231577480490524
	* Here is the AUC score for TESTING : 0.6791389109112393

	* Here is the classification report for TRAINING/TESTING :

```python
#  TRAINING set classification report 
	
              	precision    recall  f1-score   support
     0             0.37      0.77      0.50       624
     1             0.94      0.72      0.81      2875
    accuracy                           0.73      3499
   macro avg       0.66      0.75      0.66      3499
weighted avg       0.84      0.73      0.76      3499 

```

```python
#  TESTING set classification report 

             	 precision    recall  f1-score   support
           0       0.28      0.57      0.37       267
           1       0.88      0.67      0.76      1233
    accuracy                           0.66      1500
   macro avg       0.58      0.62      0.57      1500
weighted avg       0.77      0.66      0.69      1500
```


* For **Random forest Classifier** :  _rf_fit = fit_and_run(x_train, y_train, y_test,x_test,final_thune,'RandomForestClassifier')_

	* Here is the AUC score for TRAINING : 0.8864347826086957
	* Here is the AUC score for TESTING : 0.6804389889766745

	* Here is the classification report for TRAINING/TESTING : 

```python
#  TRAINING set classification report 


              precision    recall  f1-score   support
           0       0.48      0.75      0.58       624
           1       0.94      0.82      0.88      2875
    accuracy                           0.81      3499
   macro avg       0.71      0.79      0.73      3499
weighted avg       0.86      0.81      0.82      3499

```

```python
#  TESTING set classification report 

            	 precision    recall  f1-score   support
           0       0.29      0.49      0.36       267
           1       0.87      0.74      0.80      1233
    accuracy                           0.69      1500
   macro avg       0.58      0.61      0.58      1500
weighted avg       0.77      0.69      0.72      1500

```

* For **Xgboost Classifier**  : _xgb_fit = fit_and_run(x_train, y_train, y_test,x_test,final_thune,'XGBClassifier')_

	* Here is the AUC score for TRAINING 0.8514782608695652
	* Here is the AUC score for TESTING 0.6906391341723089
	* Here is the classification report for TRAINING     

```python
#  TRAINING set classification report 

          	precision    recall  f1-score   support
           0       0.39      0.80      0.52       624
           1       0.94      0.72      0.82      2875
    accuracy                           0.74      3499
   macro avg       0.66      0.76      0.67      3499
weighted avg       0.84      0.74      0.77      3499

```

```python
#  TESTING set classification report 

		precision    recall  f1-score   support
           0       0.28      0.63      0.39       267
           1       0.89      0.66      0.76      1233
    accuracy                           0.65      1500
   macro avg       0.59      0.64      0.57      1500
weighted avg       0.78      0.65      0.69      1500

```

---

### Compare all 3 model with statistical measure 

* **find_best_model_AUC()** : Output the statistics ( T-test)  related to each model  through *AUC score* , and represent a way to check wich model perform the best. This function will perform 5 split with different rand [126, 84, 42, 21, 11, 6 , 3] on the training set . For each slice , the AUC score will be save inside a dict(). At the end , i will perform a t-test through stats.ttest_rel to output the significance between the different AUC score of each model . 
 
 * output : 
	* Linear regression against XGBOOST : 
		* {'lr_vs_xgb': [Ttest_relResult(statistic=-3.896216815474171, 			 **pvalue=0.008018925191967525**)]
		
	* Linear regression against RANDOM FOREST : 
 		* 'lr_vs_rf': [Ttest_relResult(statistic=-1.7213405604456844, **pvalue=0.13597234279806697**)] 
		
	* XGBOOST against RANDOM FOREST
		* 'xgb_vs_rf': [Ttest_relResult(statistic=1.8189290553715831, **pvalue=0.11879644846260891**)]}

As depicted, while there is no difference in term of AUC score  between 
XGBOOST and random forest and between Logistic regression and RANDOMforest, the accuracy is statistically different  between LR and XGBOOST, meaning that **XGBOOST outperform LR** in term of accuracy. 

---

### Plot ROC and Learning curves  :

* **plot_roc_curves()** : Will plot the ROC curves based on StratifiedKFold with k = 10, with AUC score for each FOLD.

* **plot_learning_curves()**:  this function will check the training and testing AUC score ,  based on different training data size * Training_ratio = [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1]* .  
It represent a way to check the required number of data in order to have a good classification.

Both of this function will be run on the train data. **output_dir** need to be set. It represent the Output directory were the plot will be saved , exemple *output_dir =*'C:\Users\hamid\Servier_test_internship'*

---

### Predict P1 proprety for any given smile molecule  

In order to predict the P1 propreties for any molecule , re use the already trained model ( _xgb_fit_ , _rf_fit_ , _lr_fit_ ) for any given smile molecule , run : 

```python

# path_to_smile_dataframe_.csv need to be dataframe of lenght 2048 

 df = pd.read_csv('path_to_smile_dataframe_.csv',sep=',') 

 df['Morgan_fingerprint'] = df['smiles'].map(computeMorganFP)  # get morganFingerprint as binary array

 data = df.Morgan_fingerprint.apply(pd.Series) 
 
 # run the dataframe on the already trained model 

 if data.shape[1] != 2048 :
	print ('csv of binary feature NEED to be of  length 2048, otherwise dont work ')
 else:
 	xgb_predict = (xgb_fit.predict(data))  # XGBOOST
	print(xgb_predict)

	rf_predict = (rf_fit.predict(data))  # Random forest
	 print(rf_predict)

	lr_predict = (lr_fit.predict(data))  # Linear
 	print(lr_predict)

 ```
 
This output  will be 0 or 1  for the given binary vector ( P1 prediction  ) 




I expect to see the more complex model performing better than the simplest one, however, it won't be the case, all 3 have basically have the same accuracy. 
The preprocessing has been really fast since no NA are present, I transformed the smile representation into a binary value representing the molecule structure.

!!! There is 2 parameters to change inside the code !!!!  : 
file_dir= path to the csv file 
output_dir = path to  the directory , will be used to stock the plot

 ---

## Conclusion 

All 3 model perform in average   ~ 70 % of AUC score  as plotted in the plot_roc_curves and the minimum number of data required to output this score seems to be around 40% of the data training size. Concerning the different model used, I expected to see the more complex model XGBOOST performing better than the simplest one in a linear relation with XgoostClassifier > RandomForestClassifier > linear_model.LogisticRegression. However, it was not the case, it appear that only XgboostClassifier outperform the linear model ( sligthly ) based on the T-test .  


**The trade off between accuracy and computation time , might lead to choose the logistic regression as the main model , since it is by far the fastest model to run and perform also very well.**


## Perspective 
I also tested other method for tuning hyperparameters like _optuna_ . It might be good to check the litterature about this approach , and see if it actual outperform in term of computation time / accuracy , other more 'regular' approach 
