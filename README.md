# Music-Speech Segmentation in Bollywood 🎶

## **Problem**: I'm tired of listening to 'Dilbara' everytime I rewatch Dhoom (2004) so I've set out to create a tool that hopes to correctly identify when a song starts playing in a movie and skip it. This repository is for the purposes of tracking it's progress.

# Checklist:
1) Collect GTZAN Music-Speech dataset ✔
2) Use Librosa to extract features like MFCCs, Spectral Centroid, Zero Crossing Rate, Chroma Frequencies, Spectral Roll-off etc. ✔
3) Preprocess the dataset i.e. scale the features and encode the labels ✔
4) Train the preprocessed dataset on default ML Models such as SVMs, KNN, Random Forest, etc. ✔
5) Compare the various models and predict demo files on the best performing models ✔

# To-Do List:
1) Find a way to extract audio in real-time while a movie is playing ❌
2) Deploy the model on an entire movie and return the frames where music is detected ❌
3) Integrate it with any python library that has skip/fast-forward methods ❌
4) Can also try training the model on a deep-neural network to improve performance ❌
5) Can also try extract visual features from the dataset and train a CNN on it ❌

# Findings:
Early training on basic ML models yielded an accuracy of 95%+, some even 100%, but that shouldn't be too surprising since there are only 26 validation samples.
Out of the eight models tested, KNN, SVMs and Random Forest perform the best in terms of accuracy. However, when 5 demo files were tested, only Random Forest was able to correctly identify all of them. 

## References:
Librosa: https://librosa.org/doc/latest/index.html
Feature Extraction with Librosa: https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8
