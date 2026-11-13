"""Comprobación automática básica del proyecto académico (no clínica)."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

BASE=Path(__file__).resolve().parent
metadata=json.loads((BASE/'models'/'metadata.json').read_text(encoding='utf-8'))
model=joblib.load(BASE/'models'/'modelo.joblib')
df=pd.read_csv(BASE/'data'/'breast_cancer_wisconsin.csv')
assert len(df)==569
assert df['diagnosis'].value_counts().to_dict()=={0:357,1:212}
assert len(metadata['features'])==10
assert not df[metadata['features']].isna().any().any()
X=df[metadata['features']]; y=df['diagnosis']
_,Xtest,_,ytest=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
pred=model.predict(Xtest)
cm=confusion_matrix(ytest,pred)
assert cm.tolist()==[[71,1],[2,40]],cm.tolist()
assert list(Xtest.columns)==metadata['features']
proba=model.predict_proba(Xtest)
assert proba.shape==(114,2)
assert ((proba>=0)&(proba<=1)).all()
print('Pruebas superadas: 569 casos, 10 variables, modelo SVM, matriz [[71, 1], [2, 40]].')
