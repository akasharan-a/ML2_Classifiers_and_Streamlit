import pandas as pd
from preprocessing import preprocessing_pipeline  , drop_duplicate_rows
from sklearn.pipeline import Pipeline
import joblib

from sklearn.linear_model import LogisticRegression


df_train = pd.read_csv('data/Industrial Fabric Quality Inspection Dataset - Train.csv')

df_train = drop_duplicate_rows(df_train)

y='fabric_quality'

X_train , y_train = df_train.drop(y,axis=1) , df_train[y]

preprocessor = preprocessing_pipeline()
model = LogisticRegression(random_state=100)

model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", model)
])

algo = 'LogisticRegression'

try:
    model_pipeline.fit(X_train,y_train)
    print(f"{algo} trained successfully - ✓")
except:
    print(f"Failed to train {algo} - ✕")

try:
    joblib.dump(model_pipeline, f'model_files/{algo}.pkl')
    print("Model saved successfully - ✓")
except:
    print("Failed to save the model - ✕")