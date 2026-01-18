from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
set_config(transform_output="pandas")

##Continuous feats
cont_features = ['thread_count', 'gsm', 'tensile_strength', 'shrinkage_percent',
       'color_fastness', 'fabric_thickness', 'defect_count',
       'elongation_percent', 'moisture_absorption','machine_temperature',
       'humidity_level' ]
##Categorical features
cat_features = [ 'fabric_type','weave_type', 'finish_type', 'production_method', 'inspection_notes']
##Target variable
target = 'fabric_quality'

def drop_duplicate_rows(df):
    return df.drop_duplicates()

def preprocessing_pipeline(scale=True):
    # Continuos feats : Mean Imputation > Standard Scaling
    cont_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler() if scale else None)
    ])

    # Categorical feats : Mode Imputation > One hot encoding
    cat_preprocess = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False))
    ])

    # Combined Pipeline
    transformer = ColumnTransformer(
        transformers=[
            ("cont", cont_preprocess, cont_features),
            ("cat", cat_preprocess, cat_features)
        ],
        remainder="drop" 
    )
    
    return transformer 
