import pandas as pd
from preprocessing import preprocessing_pipeline , drop_duplicate_rows
from sklearn.pipeline import Pipeline
import joblib

from sklearn.tree import DecisionTreeClassifier

##Loading Data
df_train = pd.read_csv('data/Industrial Fabric Quality Inspection Dataset - Train.csv')

##Preparing Data
df_train = drop_duplicate_rows(df_train)

y='fabric_quality'

X_train , y_train = df_train.drop(y,axis=1) , df_train[y]
print(f"Training size : {X_train.shape[0]}\nFeatures : {X_train.shape[1]}")

##Model Pipeline
algo = 'DecisionTreeClassifier'
preprocessor = preprocessing_pipeline(scale=False)
model = DecisionTreeClassifier(random_state=100)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

##Training and Saving Model
try:
    model_pipeline.fit(X_train,y_train)
    print(f"{algo} trained successfully - ✓")
except:
    print(f"Failed to train {algo} - ✕")

try:
    joblib.dump(model_pipeline, f'model_files/{algo}.pkl')
    print("Model saved - ✓")
except:
    print("Failed to save the model - ✕")