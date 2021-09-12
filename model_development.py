
'''
Author: David Rodrigo Sánchez Navarro
Date: 12:45 GMT-5, 2020/11/21 
'''

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# ===== DATASETS
df_train = pd.read_csv(r'..\data\train.csv', encoding='utf-8')
df_test = pd.read_csv(r'..\data\test.csv', encoding='utf-8')

# ===== APPEND ALL DATA
df_train['T'] = 1
df_test['T'] = 2
df = df_train.append(df_test, ignore_index=True)

# ===== DROP FEATURES (PARA EJEMPLO)
keeps = [
    'PassengerId',
    'Survived',
    'Pclass',
    'Sex',
    'Age',
    'SibSp',
    'Parch',
    'Fare',
    'Embarked',
    'T'
]

df = df[keeps]

# ===== FEATURES AND TARGET / DATA SPLIT
from sklearn.model_selection import train_test_split

X = df[df['T']==1].drop(columns=['PassengerId', 'Survived', 'T'])
y = df[df['T']==1]['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ===== PIPELINE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, LabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn_pandas import DataFrameMapper
from sklearn.decomposition import PCA

from sklearn.metrics import accuracy_score as accs, roc_auc_score as rocas
from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Mapper
mapper = DataFrameMapper([
    (['Age', 'Fare'], [SimpleImputer(missing_values=np.nan, strategy='mean'), StandardScaler()]),
    (['SibSp', 'Parch'], None),
    (['Pclass'], OneHotEncoder()),
    (['Sex'], LabelEncoder()),
    (['Embarked'], [SimpleImputer(missing_values=np.nan, strategy='most_frequent'), LabelBinarizer()])
])

pre_steps = [('features', mapper), ('selection', PCA(n_components=.95))]

# Models
models = {
    # XGBoost model
    'XGBC': [
        ('XGBC', XGBClassifier()),
        dict(
            XGBC__max_depth=[4, 8, 12],
            XGBC__n_estimators=[10, 100, 500]
        )
    ],
    # LigtGBM model
    'LGBM': [
        ('LGBM', LGBMClassifier()),
        dict(
            LGBM__max_depth=[4, 8, 12],
            LGBM__n_estimators=[50, 100, 500]
        )
    ]
}

# Evaluation
import pickle

results = dict()

for name, items in models.items():
    print('Modelo {} en proceso...\n'.format(name))

    steps = pre_steps + [items[0]]

    pipeline = Pipeline(steps)
    model = GridSearchCV(pipeline, param_grid=items[1], cv=3)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    results[name] = {
        'scores': (round(accs(y_test, y_pred), 6), round(rocas(y_test, y_pred), 6)),
        'best_params': model.best_params_
    }

    with open(f'{name}_model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print('\nModelo {} terminado'.format(name))

print('Fin de entrenamiento y predicciones')

# ===== BEST MODEL
results


# ===== MODEL EXPORT TO DOCKER FOLDER
# Test
with open(f'XGBC_model.pkl', 'rb') as bm:
    best_model = pickle.load(bm)

new_data = {
    'Pclass': 1,
    'Sex': 'female',
    'Age': 25,
    'SibSp': 1,
    'Parch': 1,
    'Fare': 15,
    'Embarked': 'S'
}

row = pd.DataFrame(new_data, index=[0])

best_model.predict(row)[0]

# Hacia docker folder
import os
os.rename('XGBC_model.pkl', '../e18-ml-titanic-app-docker/XGBC_model.pkl')