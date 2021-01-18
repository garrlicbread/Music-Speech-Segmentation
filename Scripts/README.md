# These scripts perform the following functions:

## 1) Feature_extraction.py: Use librosa to extract the necessary features from the GTZAN Music-Speech Dataset.

## 2) Data Preprocessing.py: Scale the independant features and encode the labels

## 3) Music_speech_classification_ml.py: Train the dataset on eight default ML models. These are:
### a) Logistic Regression Classifier
### b) K Nearest Neigbors Classifier
### c) Support Vector Machine (Linear) Classifier
### d) Support Vector Machine (Gaussion) Classifier
### e) Naive Bayes Classifer
### f) Decision Tree Classifier
### g) Random Forest Classifier
### h) XG Boost Classifier

## 4) Demo_prediction.py: Train the dataset on the top performing models and test it on 5 audio files taken from the internet. These are:
### a) Dilbara.mp3 - The song that insprired this project. Good for testing True Positives.
### b) M Bole To.mp3 - A classic song from Munna Bhai M.B.B.S. Good for testing False Postives (Type 1 Error) because some parts of this song have random dialogues with no background music.
### c) Funny Dialogue.mp3 - An conversation between three characters taken from the movie Hera Pheri. Good for testing True Negatives.
### d) Sad Dialogue.mp3 - A sad dialogue depicting old-school Bollywood dialogues. Good for testing False Negatives (Type 2 Error) due to an usually loud amount of background music being played as a woman vents about something.
