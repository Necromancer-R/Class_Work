'''
Tasks
        With nulls choose a suitable imputation method for both categorical and numeric columns
        Check for outliers
        Standardize numeric features
        Encode categorical features appropriately BEFORE training
        Make sure to build imputation models/scaling on the TRAINING set only.
        Maximize F1 Scores
            Remove columns that you don't think are necessary
            Attempting different imputation methods
            Creating new features (HAVE NOT DONE YET, I think I attempted this correctly?)
            Try a different classifier (anything in Scikit-Learn)

#Wary of overfitting (making my model too complex and it's just predicting the training data),


'''

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.metrics import classification_report, f1_score
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
#There are a few extra libraries in here that I was playing with and testing things
#They didn't work out. 
'''
======
1 EDA
======
    -Explore data
        Histogram, Pairplots, Correlation Matrix
    -Examine distributions
    -Correlations

'''
I_train = pd.read_csv('hw-2-training-data.csv')
I_test = pd.read_csv('hw-2-test-data.csv')

I_train.head()
I_train.info()
I_train.describe()

#Histogram
I_train.hist(bins=50, figsize=(12,8))


#Scatterplot of systolic blood pressure
#This is the pressure in your arteries when your heart beats
#Normal is 120/80. Top number is systolic
plt.figure(figsize=(8,6))
plt.scatter(I_train['age'], I_train['bmi'], label='BMI')
plt.scatter(I_train['age'], I_train['systolic_bp'], label='S_BP')

plt.xlabel('Age')
plt.ylabel('Value')
plt.title('Age vs. BMI & BP')
plt.grid(True)
plt.legend()
plt.show()

#Scatterplot of diastolic blood pressure this is the measure between your heart beats
#Normal levels for both are 120/80. 
#bottom number is diastolic
plt.figure(figsize=(8,6))
plt.scatter(I_train['age'], I_train['bmi'], label='BMI')
plt.scatter(I_train['age'], I_train['diastolic_bp'], label='D_BP')

plt.xlabel('Age')
plt.ylabel('Value')
plt.title('Age vs. BMI & BP')
plt.grid(True)
plt.legend()
plt.show()



#Correlation Matrix
corr_matrix=I_train.corr(numeric_only=True)

plt.figure(figsize=(10,9))
sns.heatmap(corr_matrix,
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Matrix", fontsize=16)
plt.show()


'''
Using histograms to see how all columns are behaving but nothing pops out.
Decided to go with scatter plots against age to view the numerous variables against age
    1. Ran age against bmi and bp to see if anything shows. The data isn't showing any discernable pattern
    with this chart however, I can see there are outliers. I don't believe people can be negative bp
    and negative bmi because that would mean they're dead. I will need to deal with these. 
    2. Running my correlation matrix I can see that there are only a handful of variables that have correlation.
    For example with age and heart disease we see a positive 0.63 correlation so the older we get the higher number
    of people we will see that have heart disease. Then we also see a correlation of bmi and heart disease, while 
    it is only .37 it's still the second highest correlation between variables.

'''
'''

 -Identify Nulls    

'''
Inull = I_train.isnull().any(axis=1)
I_train.loc[Inull].head()

'''
================
2 Preprocessing
================
    Outliers
        Methods 
            Z-Scores, IQR.
            Things of note for this potion, the Zscores gave an empty dataframe. I interpreted this as it failed to detect any outliers.
            Not sure if that is correct or not. However, when I used IQR it detected 4 outliers in both systolic and diastolic.
            
        Dealing With
            Removal, Imputation (w/ Median)
            Did not remove NAs in this step, replaced them with imputation of medians.
    Encode features
        Choosing a categorical value to encode so it can be numerically processed
        Encoded my categorical variables of exercise frequency, diet quality, and gender. Used label encoder to encode them.
    Standardize features
        Using scaling to standardize the measurable properties of the data
            Used StandardScaler on this step.
    Handle nulls (imputation or other method)
        Imputations methods
           Imputed NAs using KNNImputer with 5 neighbors. Then after that I dropped any remaining NAs.

'''

#Outlier detection Zscores
I_train['Z_Score_Systolic'] = zscore(I_train['systolic_bp'])
I_s_outliers = I_train[np.abs(I_train['Z_Score_Systolic']) > 3]
print(I_s_outliers)

I_train['Z_Score_Diastolic'] = zscore(I_train['diastolic_bp'])
I_d_outliers = I_train[np.abs(I_train['Z_Score_Diastolic']) > 3]
print(I_d_outliers)


#4 outliers detected on systolic
Q1 = I_train['systolic_bp'].quantile(0.25)
Q3 = I_train['systolic_bp'].quantile(0.75)
IQR = Q3 - Q1
I_d_iqr = I_train[(I_train['systolic_bp'] < (Q1 - 1.5 * IQR)) | (I_train['systolic_bp'] > (Q3 + 1.5 * IQR))]
print(I_d_iqr)

#4 outliers detected on diastolic
Q1 = I_train['diastolic_bp'].quantile(0.25)
Q3 = I_train['diastolic_bp'].quantile(0.75)
IQR = Q3 - Q1
I_d_iqr = I_train[(I_train['diastolic_bp'] < (Q1 - 1.5 * IQR)) | (I_train['diastolic_bp'] > (Q3 + 1.5 * IQR))]
print(I_d_iqr)

#Imputing Outliers with Median
I_imputed = I_train.copy()
I_imputed['systolic_bp'].where(~I_train.index.isin(I_s_outliers.index), I_train['systolic_bp'].median(), inplace=True)
I_imputed['diastolic_bp'].where(~I_train.index.isin(I_d_outliers.index), I_train['diastolic_bp'].median(), inplace=True)

plt.figure(figsize=(6, 4))
plt.boxplot(I_imputed['systolic_bp'].dropna())
plt.title('Systolic BP After Outlier Imputation')
plt.grid(True)
plt.show()

plt.tight_layout()
plt.show()

#Encoder
encode = LabelEncoder()
I_imputed['Exercise Encoded'] = encode.fit_transform(I_imputed['exercise_frequency'])
I_imputed['Diet Encoded'] = encode.fit_transform(I_imputed['diet_quality'])
I_imputed['Gender Encoded'] = encode.fit_transform(I_imputed['gender'])
I_imputed = I_imputed.drop(columns=['exercise_frequency'])
I_imputed = I_imputed.drop(columns=['shoe_size'])
I_imputed = I_imputed.drop(columns=['family_history'])
I_imputed = I_imputed.drop(columns=['us_state'])
I_imputed = I_imputed.drop(columns=['diet_quality'])
I_imputed = I_imputed.drop(columns=['gender'])
print(I_imputed.head())

#Scaler
scaler = StandardScaler()
I_numeric_cols = ['bmi', 'systolic_bp', 'diastolic_bp']
I_imputed[I_numeric_cols] = scaler.fit_transform(I_imputed[I_numeric_cols]) 

#Imputing NAs KNN, and NA cleanup
I_imputed = I_imputed.drop(columns=['Z_Score_Systolic', 'Z_Score_Diastolic'])

I_knn = KNNImputer(n_neighbors=5)
I_na_array = I_knn.fit_transform(I_imputed)
I_na_imputed = pd.DataFrame(I_na_array, columns=I_imputed.columns, index=I_imputed.index)
I_imputed = I_imputed.dropna()

'''
I am adding this step below because I was running into an error with my model test. Since the training set has the column has_disease
and the test set does not, I needed to add it back in so that I could test the model and avoid dimensionality issues.
'''
X_train_impute = I_imputed.drop(columns=['has_disease'])
y_train = I_imputed['has_disease']
I_knn = KNNImputer(n_neighbors=5)
X_train_imputed_array = I_knn.fit_transform(X_train_impute)
X_train_imputed = pd.DataFrame(X_train_imputed_array, columns=X_train_impute.columns, index=X_train_impute.index)
I_na_imputed = pd.concat([X_train_imputed, y_train], axis=1)


'''
========================
3 Modeling & Evaluation
========================

    Try more than one classifier
            In this step I have chosen to use SGD and Forest classifiers. They were both very positive
        In SGD I received a total F1 score of 0.68 and from Forest a 0.85, These are fairly good scores
        Things of note, when I initially was trying to run this I kep running into an error where my I_na_imputed model
        kept throwing an error that it was not a valid input. Since KNN converts to an array, I had to bring it back to a dataframe
        in order to run the model. That took me a long time to figure out because I did not realize it was converting to an array. 
        Then obviously that's not going to run an array into a dataframe. Some other errors that I ran into, during my Z_Score step back at the start
        the way I originally had it, I was rewriting the Zscores so my entire Zscore column was all NANs due to this. So I split them
        between systolic and diastolic. Then in order to run the model I dropped them because that step wasn't necessary for the model.
    Use a cross-validation procedure
        Cross validation returned excellent results for both models.
    Avoid data leakage
        -I believe I did not have data leakage. I need to ask how to detect this in the next class.
    Correctly evaluation F1-Scores
        -F1 Score Analysis
    I have now trained my model on both SGD and Forest classifiers. My SGDClassifier returned, [0.8033, 0.5018, 0.5409, 0.8238, 0.7349]. With an average F1 of 0.68 this isn't bad. 
    However, I am getting a large variance between my folds. I see a 0.82 and a 0.50 which means there's some instability. So my model is 
    sensitive to my training data. Methods I could use to improve this would be, trying different classifiers, a different kfold such as
    stratified kfold, or even trying to tune my hyperparameters. For the sake of this assignment I will leave it as is since it meets the assignment
    requirement of 65% F1 score. But in the real world I would spend time to run my data through many different elements to see if I can improve the model.
'''

#Modeling
target = 'has_disease'

X = I_na_imputed.drop(columns=[target])
y = I_na_imputed[target].round().astype(int) 

sgd = SGDClassifier(random_state=42)
sgd_scores = cross_val_score(sgd, X, y, cv=5, scoring='f1')
print("SGDClassifier F1 scores:", sgd_scores)
print("SGDClassifier Mean F1:", sgd_scores.mean())

rf = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=5, scoring='f1')
print("RandomForestClassifier F1 scores:", rf_scores)
print("RandomForestClassifier Mean F1:", rf_scores.mean())

#Evaluation
I_scores_sgd = cross_validate(sgd, X, y, cv=5, scoring='f1')
print("SGDClassifier Cross-Validation F1 scores:", I_scores_sgd['test_score'])

I_scores_forest = cross_validate(rf, X, y, cv=5, scoring='f1')
print("RandomForestClassifier Cross-Validation F1 scores:", I_scores_forest['test_score'])


'''
============
4 Test Data
============
    Train your model appropriately
        Successfully matched test set to training set. There was a dimensionality issue that I had to resolve with the
        has_disease column. Everything performed correctly.
    Impute data correctly
    Achieve at least 65% F1 on the test data
    Save your answers to an answers.csv file
    Answers saved

'''
#Matching test data to training data. All steps that were done to the training data will be done to the test data.
I_test_processed = I_test.copy()
I_test_processed['systolic_bp'].where(
    I_test_processed['systolic_bp'].between(
        I_train['systolic_bp'].quantile(0.25),
        I_train['systolic_bp'].quantile(0.75)
    ),
    I_train['systolic_bp'].median(),
    inplace=True
)
I_test_processed['diastolic_bp'].where(
    I_test_processed['diastolic_bp'].between(
        I_train['diastolic_bp'].quantile(0.25),
        I_train['diastolic_bp'].quantile(0.75)
    ),
    I_train['diastolic_bp'].median(),
    inplace=True
)
exercise_encoder = LabelEncoder()
diet_encoder = LabelEncoder()
gender_encoder = LabelEncoder()
exercise_encoder.fit(I_train['exercise_frequency'])
diet_encoder.fit(I_train['diet_quality'])
gender_encoder.fit(I_train['gender'])

I_test_processed['Exercise Encoded'] = exercise_encoder.transform(I_test_processed['exercise_frequency'])
I_test_processed['Diet Encoded'] = diet_encoder.transform(I_test_processed['diet_quality'])
I_test_processed['Gender Encoded'] = gender_encoder.transform(I_test_processed['gender'])

I_test_processed = I_test_processed.drop(columns=[
    'exercise_frequency', 'shoe_size', 'family_history', 'us_state', 'diet_quality', 'gender'
])

print(I_test_processed.head())
I_test_processed[I_numeric_cols] = scaler.transform(I_test_processed[I_numeric_cols])

I_test_processed_array = I_knn.transform(I_test_processed)
I_test_imputed = pd.DataFrame(I_test_processed_array, columns=I_test_processed.columns, index=I_test_processed.index)

#Final Model Training and Evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print(classification_report(y_test, y_pred))

#Final Model Testing
final_model = RandomForestClassifier(random_state=42)
final_model.fit(X, y)
test_prediction = final_model.predict(I_test_imputed)
output_hw_df = pd.DataFrame({'Prediction': test_prediction})
output_hw_df.to_csv('answers.csv', index=False)
print("✅ Predictions saved to 'answers.csv'")

# %%
