# Now that we have the extracted features in a .csv file, lets pre-process the data to apply ML Models

# Importing libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Importing the dataset
path = "C://Desktop/Python/Projects/Skip music"
df = pd.read_csv("C://Desktop/Python/Projects/Skip music/music_speech_features.csv")
X = df.iloc[:, : -1].values
y = df.iloc[:, -1]
columns = df.iloc[:, : -1].columns

# Scaling the features
sc = StandardScaler()
X[:, 1 : ] = sc.fit_transform(X[:, 1 : ])

# Encoding the dependant variable
le = LabelEncoder()
y = le.fit_transform(y)

# Saving the preprocessed df into a new .csv file
X = pd.DataFrame(X)
y = pd.DataFrame(y)
preprocessed_dataframe = pd.concat(objs = (X, y), axis = 1)
preprocessed_dataframe.columns = columns
preprocessed_dataframe.to_csv("Preprocessed_Dataset.csv")
