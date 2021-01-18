# Detecting a single file (Dilbara.wav)

# NOTE: 
# Even though we've aready pre-processed the data, 
# We'll do it again in this demo script so that we can use sc.transform to standardize the demo file

# Initializing starting time
import time
start = time.time()

# Ignoring warnings
import warnings
warnings.filterwarnings('ignore')

# Importing libraries 
import librosa
import numpy as np
import pandas as pd 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Initializing paths
df_path = "C:/Users/Sukant Sidnhwani/Desktop/Python/Projects/Bollywood Music Skipper/music_speech_features.csv"
demo_file_path = "D:/Big Datasets/Movie & Music/"

# Training the four main models on the dataset
df = pd.read_csv(df_path)
del df['Filename']
X = df.iloc[:, : -1].values
y = df.iloc[:, -1]
columns = df.columns

# Scaling the features
sc = StandardScaler()
X = sc.fit_transform(X)

# Encoding the dependant variable
le = LabelEncoder()
y = le.fit_transform(y)

# Saving the preprocessed df into a new .csv file
X = pd.DataFrame(X)
y = pd.DataFrame(y)
preprocessed_dataframe = pd.concat(objs = (X, y), axis = 1)
preprocessed_dataframe.columns = columns

X = preprocessed_dataframe.iloc[:, : -1].values
y = preprocessed_dataframe.iloc[:, -1].values

# Splitting the df into training/test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, shuffle = True, random_state = 42)

# KNN Classifier
scorelist = []
scores = pd.DataFrame()
for i in range(1, 26):
    knn_cl = KNeighborsClassifier(n_neighbors = i)
    knn_cl.fit(X_train, y_train)
    y_pred = knn_cl.predict(X_test)
    ac1 = accuracy_score(y_test, y_pred)
    scorelist.append(ac1)
ac1 = sorted(scorelist, reverse = True)[:1]
ac1 = float(np.array(ac1))
scores['Accuracies'] = scorelist
kindex = int((scores.idxmax() + 1))
knn_cl = KNeighborsClassifier(n_neighbors = kindex).fit(X_train, y_train)

# Support Vector Machine model [Linear]
svm_cl = SVC(kernel = 'linear').fit(X_train, y_train)
y_pred2 = svm_cl.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)

# Support Vector Machine model [Gaussian]
svmg_cl = SVC(kernel = 'rbf').fit(X_train, y_train)
y_pred3 = svmg_cl.predict(X_test)
ac3 = accuracy_score(y_test, y_pred3)

# Random Forest
rflist = []
rscores = pd.DataFrame()
for i in range(1, 200):
    rf_cl = RandomForestClassifier(n_estimators = i, criterion = 'entropy')
    rf_cl.fit(X_train, y_train)
    y_pred4 = rf_cl.predict(X_test)
    rf_accuracy = accuracy_score(y_test, y_pred4)
    rflist.append(rf_accuracy)
ac4 = sorted(rflist, reverse = True)[:1]
ac4 = float(np.array(ac4))
rscores['Accuracies'] = rflist
rindex = int((rscores.idxmax() + 1))
rf_cl = RandomForestClassifier(n_estimators = rindex, criterion = 'entropy').fit(X_train, y_train)
    
# Evaluating all accuracies into a list
list1 = [ac1, ac2, ac3, ac4]
list1 = [i * 100 for i in list1]
list1 = [round(num, 2) for num in list1]

# Appending these accuracies in a DataFrame
demo = pd.DataFrame()
demo['Classification Models'] = ['KNN Classification', 
                                 'SVM [Linear] Classification', 
                                 'SVM [Gaussian] Classification',
                                 'Random Forest Classification']
demo['Accuracies'] = list1

# Printing the final dataframe
print()
print(demo)
print()
        
# Extracting features from the demo file and appending it to a demo dataframe
while True:
    audiofile = input("Enter name of the file: ")
    if audiofile.lower() == "exit" or audiofile.lower() == "break":break
    else:
        def audiophile(x):
            xy, sr = librosa.load(demo_file_path + x, mono = True)
            return xy, sr
        xy, sr = audiophile(audiofile)
        cf = librosa.feature.chroma_stft(y = xy, sr = sr)
        spc = librosa.feature.spectral_centroid(y = xy, sr = sr)
        sb = librosa.feature.spectral_bandwidth(y = xy, sr = sr)
        spr = librosa.feature.spectral_rolloff(y = xy, sr = sr)
        zc = librosa.feature.zero_crossing_rate(y = xy)
        mfccs = librosa.feature.mfcc(y = xy, sr = sr)
        rmse = librosa.feature.rms(y = xy)
        feature_list = [np.mean(cf), np.mean(spc), np.mean(sb), np.mean(spr), np.mean(zc), np.mean(rmse)]
        for i in mfccs:
            feature_list.append(np.mean(i))
        demo_df = pd.DataFrame(feature_list)
        demo_df = demo_df.transpose()
        
        # Preprocessing the extracted features
        unscaled_df = np.array(demo_df, dtype = 'object')
        scaled_df = sc.transform(unscaled_df)
        
        # Predicting the demo file and appending the predictions to demo df
        demo1 = knn_cl.predict(demo_df)
        demo2 = svm_cl.predict(demo_df)
        demo3 = svmg_cl.predict(demo_df)
        demo4 = rf_cl.predict(demo_df)
        demo_list2 = [demo1, demo2, demo3, demo4]
        demo_list2 = ["Speech" if i == 1 else "Music" for i in demo_list2]
        demo['Demo Predictions'] = demo_list2
        
        # Printing the final dataframe
        print()
        print(demo)
        print()
        
        # # Initiating ending time
        # end = time.time()
        # print()
        # print(f"This program executes in {round((end - start), 2)} seconds.")
