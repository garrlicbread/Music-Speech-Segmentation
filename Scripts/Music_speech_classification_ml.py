# Now that we have the pre-processed data, we're ready to apply various models on it

# Initializing starting time
import time
start = time.time()

# Importing Libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Importing data and assigning variables
dataset = pd.read_csv('C://Desktop/Python/Projects/Skip music/Preprocessed_Dataset.csv')
del dataset['Filename']

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:, -1].values

# # Visualizing the correlations
# corr = dataset.corr()
# sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns)

# Splitting the dataset into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True)
# y_train = y_train.reshape(-1, 1).reshape(-1)

# Fitting a logistic regression model 
log_cl = LogisticRegression(max_iter = 10000).fit(X_train, y_train)
y_pred1 = log_cl.predict(X_test)

ac1 = accuracy_score(y_test, y_pred1)

# Fitting a K nearest neighbours model 

# Default method i.e. manually choose the number of neighbours 
# knn_cl = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2).fit(X_train, y_train)
# y_pred2 = knn_cl.predict(X_test)

# ac2 = accuracy_score(y_test, y_pred2)

# Iterative method i.e. run a loop of different neighbours and choose the best one 
scorelist = []
scores = pd.DataFrame()

for i in range(1, 26):
    knn_cl = KNeighborsClassifier(n_neighbors = i)
    knn_cl.fit(X_train, y_train)
    y_pred2 = knn_cl.predict(X_test)
    ac2 = accuracy_score(y_test, y_pred2)
    scorelist.append(ac2)

ac2 = sorted(scorelist, reverse = True)[:1]
ac2 = float(np.array(ac2))
scores['Accuracies'] = scorelist
kindex = int((scores.idxmax() + 1))
knn_cl = KNeighborsClassifier(n_neighbors = kindex).fit(X_train, y_train)

# Fitting Support Vector Machine model [Linear]
svm_cl = SVC(kernel = 'linear').fit(X_train, y_train)
y_pred3 = svm_cl.predict(X_test)

ac3 = accuracy_score(y_test, y_pred3)

# Fitting Support Vector Machine model [Gaussian]
svmg_cl = SVC(kernel = 'rbf').fit(X_train, y_train)
y_pred4 = svmg_cl.predict(X_test)

ac4 = accuracy_score(y_test, y_pred4)

# Fitting Naive Bayes Model
nb_cl = GaussianNB().fit(X_train, y_train)
y_pred5 = nb_cl.predict(X_test)

ac5 = accuracy_score(y_test, y_pred5)

# Fitting a Decision tree model 
dt_cl = DecisionTreeClassifier(criterion = 'entropy').fit(X_train, y_train)
y_pred6 = dt_cl.predict(X_test)

ac6 = accuracy_score(y_test, y_pred6)

# Fitting a Random Forrest model with an iterative loop 
rflist = []
rscores = pd.DataFrame()
for i in range(10, 200):
    rf_cl = RandomForestClassifier(n_estimators = i, criterion = 'entropy')
    rf_cl.fit(X_train, y_train)
    y_pred7 = rf_cl.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred7)
    rflist.append(rf_accuracy)
ac7 = sorted(rflist, reverse = True)[:1]
ac7 = float(np.array(ac7))
rscores['Accuracies'] = rflist
rindex = 10 + int((rscores.idxmax() + 1))
rf_cl = RandomForestClassifier(n_estimators = rindex, criterion = 'entropy').fit(X_train, y_train)
    
# Fitting a Gradient Boost with an iterative loop 
gblist = []
gbscores = pd.DataFrame()

for i in range(5, 200):
    gb_cl = XGBClassifier(learning_rate = 0.01, n_estimators = i)
    gb_cl.fit(X_train, y_train)
    y_pred8 = gb_cl.predict(X_test)
    gb_accuracy = accuracy_score(y_test, y_pred8)
    gblist.append(gb_accuracy)
ac8 = sorted(gblist, reverse = True)[:1]
ac8 = float(np.float64(ac8))
gbscores['Accuracies'] = gblist
gbindex = 5 + int((gbscores.idxmax() + 1))
gb_cl = XGBClassifier(learning_rate = 0.01, n_estimators = gbindex).fit(X_train, y_train)

# Evaluating all accuracies into a list
list1 = [ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8]
list1 = [i * 100 for i in list1]
list1 = [round(num, 2) for num in list1]

classification = pd.DataFrame()
classification['Classification Models'] = ['Logistic Regression Classification', 
                                           'K Nearest Neighbors Classification', 
                                           'Support Vector Machine [Linear] Classification', 
                                           'Support Vector Machine [Gaussian] Classification',
                                           'Naive Bayes Classification',
                                           'Decision Tree Classification', 
                                           'Random Forrest Classification',
                                           'XGBoost Classification']

classification['Accuracies'] = list1

# Initiating ending time
end = time.time()
print()
print(f"This program executes in {round((end - start), 2)} seconds.")
print()

print(classification)

## Saving the classification report to a .csv file
# classification.to_csv('ML_model_accuracies.csv')

# Printing the best performing models
best_m = classification.loc[classification['Accuracies'] >= 90]

print()
print(f"The best performing models are: \n\n{best_m}")
