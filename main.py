import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from functions import *
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df1 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_primary_cohort.csv")
df2 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_study_cohort.csv")
df3 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_validation_cohort.csv")
#combined all the data together
df_combined = pd.concat([df1, df2, df3], ignore_index=True)
df_combined.drop(columns=["episode_number"])

#checked the distribution of all the classes
# plt.rcParams.update({'font.size': 16})
# df_combined.hist(figsize=(12,8))
# plt.savefig('histograms.png',facecolor='white',bbox_inches="tight")

#outcomes are not balanced so we need to balance it
X = df_combined.drop(columns=['hospital_outcome_1alive_0dead'])
y = df_combined['hospital_outcome_1alive_0dead']
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


#split dataset
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': [None, 'balanced', 'balanced_subsample']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, scoring='precision', cv=5)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
evaluateModel(best_rf, X_test, y_test)