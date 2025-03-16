import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
from functions import *
from sklearn.model_selection import GridSearchCV

df1 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_primary_cohort.csv")
df2 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_study_cohort.csv")
df3 = pd.read_csv("data/s41598-020-73558-3_sepsis_survival_validation_cohort.csv")
#combined all the data together
df_combined = pd.concat([df1, df2, df3], ignore_index=True)

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
    'max_depth': [2, 3],  
    'min_samples_split': [30, 50, 70],  
    'min_samples_leaf': [30, 40, 50],  
    'ccp_alpha': [0.01, 0.05, 0.1, 0.2] 
}


grid_search = GridSearchCV(tree.DecisionTreeClassifier(random_state=42, class_weight='balanced'), param_grid, cv=5, scoring='precision')
grid_search.fit(X_train, y_train)

best_dt = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)
show_tree(best_dt)

evaluateModel(best_dt, X_test, y_test)
