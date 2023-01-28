# servier_technical_test



For this Technical test , since it is a binary classification problem , i decide  to implement 3 model , from the most basic to the more "complex"
The different model are : - linear_model.LogisticRegression
			  - RandomForestClassifier 
	                  - XGBClassifier   

I expect to see the more complex model performing better than the simplest one, however, it won't be the case, all 3 have basically have the same accuracy. 
The preprocessing has been really fast since no NA are present, I transformed the smile representation into a binary value representing the molecule structure.

!!! There is 2 parameters to change inside the code !!!!  : 
file_dir= path to the csv file 
output_dir = path to  the directory , will be used to stock the plot


Get best parameters:I want to get the best hyper parameters for each models 
inside the function   'run_all_thunes' , i will perform 3 'run'  with 3 differents slice of training data ( rand 126, 84, 42 ) , since if i rely on only on slice , i might get something 
who advantage the model, it constitute a biais in my opinion . 
The function will run :

- tune_xgboost : That function will output XGBOOST model with best parameters by performing randomgridsearch with stratified 3-fold .i rely on stratifiedkfold since the data are unbalance.
I use Randomgridsearch since based on litterature, it perform good , and might even be better than the regular gridsearch, for a high number of itteration . I set itteration to 50 since it 
also have been show to get good result . I also put scale_pos_weight with my ratio of label since it's unbalance . 

- tune_lr : That function will output linear model with the best parameters it basically act the same as tune xgboost , weight have been set to 'balance' once again to took
care of unbalance label 

- tune_rf: That function will output RandomForest model with the best parameters, it will act the same as other tune function. 

 Each 3 function will be run for the 3 slice ( rand), and be add into a dictionnary . The output of this function  run_all_thunes,  will be the best parameters for all 3 model given 
the 3 different run . 



The function 'find_best_model_AUC' ==> Allow us to know which model perform the best based on statistical significance
Inside the function   'find_best_model_AUC'   , i  will perform 5 split with different rand (126, 84, 42, 21, 11, 6 , 3 ) on the training set . for each slice , 
the AUC score will besave . At the end , i will perform a t-test through stats.ttest_rel to output the significance between the different AUC score of each model . 

output : 

{'lr_vs_xgb': [Ttest_relResult(statistic=-3.896216815474171, pvalue=0.008018925191967525)],
 'lr_vs_rf': [Ttest_relResult(statistic=-1.7213405604456844, pvalue=0.13597234279806697)],
 'xgb_vs_rf': [Ttest_relResult(statistic=1.8189290553715831, pvalue=0.11879644846260891)]}

as we can see while there is no difference in term of AUC score ( accuracy of the model) between 
XGBOOST and random forest and between Logistic regression and RANDOMforest, the accuracy is statistically different 
between LR and XGBOOST, meaning that XGBOOST outperform LR . 



Function 'fit_and_run'  ==> train and evaluate the model : 
 
Inside the function  'fit_and_run' , the best parameters for each 3 model of the function 'run_all_thunes' will be run in order to test the AUC score and 
classification report  for the initial training set and the initial testing set , in order to check if  it overfit or not . Here is the output : 

For Linear regression : 
*** Here is the AUC score for TRAINING 0.8231577480490524
*** Here is the AUC score for TESTING 0.6791389109112393

*** Here is the classification report for TRAINING 

              precision    recall  f1-score   support
           0       0.37      0.77      0.50       624
           1       0.94      0.72      0.81      2875
    accuracy                           0.73      3499
   macro avg       0.66      0.75      0.66      3499
weighted avg       0.84      0.73      0.76      3499

*** Here is the classification report for TESTING 

              precision    recall  f1-score   support
           0       0.28      0.57      0.37       267
           1       0.88      0.67      0.76      1233
    accuracy                           0.66      1500
   macro avg       0.58      0.62      0.57      1500
weighted avg       0.77      0.66      0.69      1500



For Random forest : 
*** Here is the AUC score for TRAINING 0.8864347826086957
*** Here is the AUC score for TESTING 0.6804389889766745

*** Here is the classification report for TRAINING 

              precision    recall  f1-score   support
           0       0.48      0.75      0.58       624
           1       0.94      0.82      0.88      2875
    accuracy                           0.81      3499
   macro avg       0.71      0.79      0.73      3499
weighted avg       0.86      0.81      0.82      3499

*** Here is the classification report for TESTING  

             precision    recall  f1-score   support
           0       0.29      0.49      0.36       267
           1       0.87      0.74      0.80      1233
    accuracy                           0.69      1500
   macro avg       0.58      0.61      0.58      1500
weighted avg       0.77      0.69      0.72      1500


For Xgboost :
*** Here is the AUC score for TRAINING 0.8514782608695652
*** Here is the AUC score for TESTING 0.6906391341723089

*** Here is the classification report for TRAINING     

          precision    recall  f1-score   support
           0       0.39      0.80      0.52       624
           1       0.94      0.72      0.82      2875
    accuracy                           0.74      3499
   macro avg       0.66      0.76      0.67      3499
weighted avg       0.84      0.74      0.77      3499

*** Here is the classification report for TESTING 

              precision    recall  f1-score   support
           0       0.28      0.63      0.39       267
           1       0.89      0.66      0.76      1233
    accuracy                           0.65      1500
   macro avg       0.59      0.64      0.57      1500
weighted avg       0.78      0.65      0.69      1500


Function plot_learning_curves ==>  this function will check the training and testing AUC score ,  based on the training data size . 
That way , we will know if this model perform better with huge number of data or not . 

Function  plot_roc_curves ==> This function will plot the ROC curve based on 10 fold .


finally to predict for any given smile molecule , run : 

# 1) transform smile dataframe into binary vector 

 df = pd.read_csv('path_to_smile_dataframe_.csv',sep=',')

 df['Morgan_fingerprint'] = df['smiles'].map(computeMorganFP)  # get morganFingerprint as binary array

 data = df.Morgan_fingerprint.apply(pd.Series) 


# run the dataframe on the already trained model 
   
 xgb_predict = (xgb_fit.predict(data)) # XGBOOST
    print(xgb_predict)

    rf_predict = (rf_fit.predict(data )) # Random forest
    print(rf_predict)

    lr_predict = (lr_fit.predict(data )) # Linear
    print(lr_predict)

with x_test being a dataframe of lenght 2048 , and represent a sucession of binary value . The output will be 0 or 1 ( P1 prediction  ) 


Conclusion : 
All 3 model perform in average   ~ 70 % AUC score  as plotted in the plot_roc_curves . While there is no difference between logistic regression and random forest , 
it appear that xgboost perform slightly better based on the T-test . 
However the trade off between accuracy and computation time , might lead to choose the logistic regression as the main model , since it is by far faster to perform
searching of hyperparameters and to train. 

