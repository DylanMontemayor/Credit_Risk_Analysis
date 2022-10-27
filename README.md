# Credit_Risk_Analysis

## Overview of the Module 17 Challenge

For this project, we analyzed the credit database from LendingClub company. Credit risk is an inherently unbalanced classification problem. Generally, the number of good loans is much higher than the number of risky loans. Therefore, we used imbalanced-learn and scimitar-learn machine learning libraries to train and evaluate the models to get more accurate predictions. 

### Files and Folders

credit_risk_resampling.ipynb

credit_risk_ensemble.ipynb

Resources: images

## Results

For better results, we applied some functions to clean the data before using it. We run six machine learning models to train, test, and evaluate the database. For consistency, we used random_state=1 for all the models. 

### 1. Naive Random Oversampling

The naive random model results show that it is more sensitive than it is precise. In this case, we prefer to have a higher sensitivity since we want to catch as many negatives as possible. The second most important thing to consider is that precision for low-risk loans is very high (almost 1), which will lead to have many predictions of high-risk loans that were actually low-risk. Considering the high sensitivity and very low precision, the F1 for high-risk loans ends up with a very low result.

!['One'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/One.png)

### 2.  SMOTE Oversampling

For the SMOTE Oversampling approach, we can see that even when the sensitivity for the low-risk loans improved (from 0.59 to 0.69), the overall result stayed very similar to the naive random oversampling. The f1 score for this model remained the same at 0.02. Most balanced in sensitivity

!['Two'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/Two.png)

### 3. Undersampling

When doing undersampling, we can see how in the classification report, the precision percentage of the high (0.01) and low-risk (almost 1) loans stayed almost the same. In the confusion matrix, we can see how the number of high-risk predictions for the actually low-risk ones doubled from the previous method. Even when we prefer a high sensitivity, it will still take many resources to verify all this data. 

!['Three'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/Three.png)

### 4. Combination (Over and Under) Sampling

For the SMOTEENN model that combines over and under sampling, the results didn't that much from the previous models. 

The main difference is that this model increased a little bit the sensitivity (high-risk 0.70, low-risk 0.58) but it is still lower than the navie random oversampling (high-risk 0.72, low-risk 0.59)



!['Four'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/Four.png)

### 5. Balanced Random Forest Classifier

The results of this model that reduces bias improved compared to all the under and over-sampling. It passed from having between 5k to 10k false positives to 2k. The sensitivity of the high-risk loans performed almost the same as the others (0.70) but the low-risk sensitivity improved a lot from around 0.5 to 0.87. This will reduce the resources used for verifying all these cases that were predicted wrong. 

!['Five'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/Five.png)

### 6. Easy Ensemble AdaBoots Classifier

All the results improved a lot in this model compared to the other ones. It got the highest balanced accuracy score of 0.93. The sensitivity of both low and high-risk loans improved up to over 0.90. This model achieved to catch many high-risk loans without increasing the number of false positives. 

!['Six'](https://github.com/DylanMontemayor/Credit_Risk_Analysis/blob/main/Resources/Six.png)

### Summary

For the under and over-sampling methods applied to this database, the results weren't very different from each other. Even though neither of them is a good model, the naive random and the SMOTE oversampling performed better that the other ones. 

Overall, the easy ensemble classifier was the best model performer. We can say that this model accomplishes reducing the bias of a very unbalanced database. 
