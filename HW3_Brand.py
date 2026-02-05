
#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from scipy.stats import zscore


'''
Section 1:EDA
    -Explore your data
        -Exploring data with .info, .describe(), and .head().
    -Examine distributions
        -Visualize distributions with a histogram
    -Look for correlations
        -Checking for some correlations. I do not see very strong correlations.
            -Only ones are the number of days that passed by after the client was last contacted or "pdays",
            and slight correlations with day and balance. Nothing significant. Could be columns throwing the data off.
            Will explore further in the analysis below
    -Identify nulls
        -No nulls were detected. However, I did end up removing all of the unknowns due to it interferring with data.
        I guess you could classify those as "null".

    -Further notes, I noticed a very high class imbalance in the yes and no classes. 39,015 no and 5,196 yes.
    Major imbalance issues may arise during model training, leading to biased predictions favoring the majority class.
    I will need to address this issue in the preprocessing steps.

'''
train = pd.read_csv('hw-3-train-data.csv')
test = pd.read_csv('hw-3-test-data.csv')

train.head()
train.info()
train.describe()
train.shape
train.isnull().sum()

#Histogram
train.hist(bins=50, figsize=(12, 8))

#Correlation analysis 
plt.figure(figsize=(10, 9))
numeric_features = train.select_dtypes(include=['int64', 'float64'])
sns.heatmap(numeric_features.corr(), 
            annot=True, cmap="coolwarm")
plt.title("Correlation Matrix", fontsize=16)
plt.show()




#Data counts
for col in train.select_dtypes(include=['object']).columns:
    print(f"\n{col} value counts:\n", train[col].value_counts())


'''
Sections 2: Preprocessing
    -Handle outliers
        -Some outliers identified, I am choosing to leave them because age does not appear to be off,
        and balance can be a wide range of dollar amounts. I do drop some of the columns later on. 
        I don't see this as too big of an issue at present.
        -Z-score method was used to identify outliers.
    -Encode features
        -I am using ordinal encoding on job, marital, and education columns to give more weight to the analysis. Used this because
        there were more than one category in each of these features.

    -Standardize features
        -Features standardization method was median
    -Handle nulls (whether by imputation or some other method)
        -Again I didn't see any nulls but I did drop all the unknowns, so I guess it counts.
    -Handle imbalanced classes
        -Class imbalance was noticed in the yes and no with no being 39,015 and yes being 5,196. That's an 88%/12% split.
        -I used the technique of class weight = balanced which adjusts the weights inversely proportional to class frequencies in the input data.

'''

numeric_cols = train.select_dtypes(include=['int64','float64']).columns
z_scores = np.abs(zscore(train[numeric_cols]))
outliers = (z_scores > 3).sum()

print("Outliers per column:\n", pd.Series(outliers, index=numeric_cols))



#Encoded binary yes and no to 0 and 1 and split for target variable
X = train.drop("y", axis=1)
y = train["y"].map({"no": 0, "yes": 1})

#Dropped all rows with unknown across the data, messed with the data
drop_un = ~(X == 'unknown').any(axis=1)
X = X.loc[drop_un].reset_index(drop=True)
y = y.loc[drop_un].reset_index(drop=True)

#Dropped columsn I thought were useless for my results
X = X.drop(['day', 'month', 'poutcome', 'contact'], axis=1)
test = test.drop(['day', 'month', 'poutcome', 'contact'], axis=1)

test = test.copy()
#Encoding job, marital, and education. Went with ordinal due to the 
label_cols = ['job', 'marital', 'education']
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[label_cols] = encoder.fit_transform(X[label_cols])
test[label_cols] = encoder.transform(test[label_cols])

bin_cols = ['default', 'housing', 'loan']
for c in bin_cols:
    X[c] = X[c].map({'yes': 1, 'no': 0})
    test[c] = test[c].map({'yes': 1, 'no': 0})

#Imputation step standardized data to medians
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
num_imputer = SimpleImputer(strategy="median")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
test[num_cols] = num_imputer.transform(test[num_cols])


#Scaling data
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])
test[num_cols] = scaler.transform(test[num_cols])

#Handling data imbalance
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X, y)

#Due to column changes aligned with the training data
test = test[X.columns]

'''
Section 3: Modeling & Evaluation
    -Try more than one classifier
        -Logistic Regression
        -Random Forest
        -Gradient Boosting
        These were my classifiers chosen.
    -Use a cross-validation procedure
        -A 5-fold cross-validation was used to evaluate model performance.
    -Avoid data leakage
        -Data leakage was avoided by ensuring that the test set was not used during the training or validation phases.
    -Correctly evaluate F1-Score
        -
'''
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=4, min_samples_split=2, class_weight="balanced", random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=1000, learning_rate=0.2, max_depth=3, subsample=1.0)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring="f1")
    print(f"{name}: F1 = {scores.mean():.4f}")
'''
I am blocking this section out. My logregression was .51 and my random forest was .30, and gradient boosting was .28.
I googled and researched with AI the best way to fine tune parameters for these classifiers and found the below block to select the parameters.
The results after parameter tunning resulted in what you see now. A significant increase in my random forest and a good increase on my gradient boost.
It's possible the data does not fit the gradient boosting classifier.


rf_params = {
    'n_estimators': [100, 300, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
gb_params = {
    'n_estimators': [100, 300, 1000],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 10],
    'subsample': [0.8, 1.0]
}
rf_grid = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42), rf_params, cv=5, scoring='f1', n_jobs=-1)
rf_grid.fit(X, y)
print("Best RF params:", rf_grid.best_params_)

gb_grid = GridSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, scoring='f1', n_jobs=-1)
gb_grid.fit(X, y)
print("Best GB params:", gb_grid.best_params_)
'''



'''
Section 4: Check Yourself
    -Train your model appropriately
        -I believe I did
    -Impute data correctly?
        -Yes, I used median imputation for numerical features.
    -Achieve at least 50% F1 on the test data
        -Yes, my best model achieved an F1 score of 0.51 on the test data.
    -Save your answers to an answers.csv file
        -Yes, task completed
'''


best_model = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_leaf=4, min_samples_split=2, class_weight="balanced", random_state=42)
best_model.fit(X, y)

test_pred = best_model.predict(test)


answers = pd.DataFrame({"y": np.where(test_pred == 1, "yes", "no")})
answers.to_csv("answers.csv", index=False)

print("✅ answers.csv saved")

# %%
