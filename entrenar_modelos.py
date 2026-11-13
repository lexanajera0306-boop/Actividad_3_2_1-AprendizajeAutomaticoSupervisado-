"""Reentrena cuatro modelos y regenera datos, métricas, artefactos reproducibles."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
BASE=Path(__file__).resolve().parent
for name in ("data","models"):(BASE/name).mkdir(exist_ok=True)
bc=load_breast_cancer(as_frame=True)
features=list(bc.feature_names[:10]); X=bc.frame[features]; y=1-bc.target
csv=X.copy();csv['diagnosis']=y.to_numpy();csv.to_csv(BASE/'data'/'breast_cancer_wisconsin.csv',index=False)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
configs={
'SVM':(Pipeline([('scaler',StandardScaler()),('model',SVC(probability=True,random_state=42))]), {'model__C':[.1,1,10,100],'model__gamma':['scale',.001,.01,.1], 'model__kernel':['rbf']}),
'KNN':(Pipeline([('scaler',StandardScaler()),('model',KNeighborsClassifier())]), {'model__n_neighbors':[3,5,7,9,11],'model__weights':['uniform','distance']}),
'Random Forest':(RandomForestClassifier(random_state=42,n_jobs=1), {'n_estimators':[100,200],'max_depth':[None,5,10],'min_samples_leaf':[1,2]}),
'Árbol de Decisión':(DecisionTreeClassifier(random_state=42), {'max_depth':[None,3,5,7],'min_samples_leaf':[1,2,4],'criterion':['gini','entropy']})
}
rows=[]; rep=[]; best={}
for name,(model,grid) in configs.items():
    search=GridSearchCV(model,grid,cv=cv,scoring='f1',n_jobs=-1).fit(Xtr,ytr)
    pred=search.predict(Xte); proba=search.predict_proba(Xte)[:,1]
    best[name]=search.best_estimator_
    rows.append({'Modelo':name,'F1_CV':search.best_score_,'Accuracy':accuracy_score(yte,pred),'Precision':precision_score(yte,pred),'Recall':recall_score(yte,pred),'F1':f1_score(yte,pred),'ROC_AUC':roc_auc_score(yte,proba),'Parametros':str(search.best_params_)})
    scores=cross_validate(search.best_estimator_,Xtr,ytr,cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=42),scoring={'f1':'f1','recall':'recall','precision':'precision'},n_jobs=-1)
    rep.append({'Modelo':name,'F1_medio':scores['test_f1'].mean(),'F1_std':scores['test_f1'].std(),'Recall':scores['test_recall'].mean(),'Precision':scores['test_precision'].mean()})
results=pd.DataFrame(rows);results.to_csv(BASE/'data'/'comparacion_modelos.csv',index=False)
pd.DataFrame(rep).to_csv(BASE/'data'/'validacion_repetida.csv',index=False)
selected=best['SVM']; joblib.dump(selected,BASE/'models'/'modelo.joblib')
ranges={f:{'min':float(Xtr[f].min()),'max':float(Xtr[f].max()),'mediana':float(Xtr[f].median())} for f in features}
r=results.set_index('Modelo').loc['SVM']
metadata={'modelo':'SVM','features':features,'rangos':ranges,'clases':{'0':'Benigno','1':'Maligno'},'metricas_prueba':{'Accuracy':float(r.Accuracy),'Precision':float(r.Precision),'Recall':float(r.Recall),'F1':float(r.F1),'ROC AUC':float(r.ROC_AUC)},'version_sklearn':sklearn_version}
(BASE/'models'/'metadata.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf8')
print(results[['Modelo','F1_CV','Accuracy','Precision','Recall','F1','ROC_AUC']].round(4).to_string(index=False))
