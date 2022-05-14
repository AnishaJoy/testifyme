import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import ensemble
import pickle
df=pd.read_csv("coronadataset.csv")
df = df.dropna()
for col in df:
    df.drop(df.index[df[col ] == 'None'], inplace = True)
df.drop(df.index[df['corona_result' ] == 'other'], inplace = True)
df= df.drop('test_date',axis=1)
df['cough']=df['cough'].astype(int)
df['fever']=df['fever'].astype(int)
df['sore_throat']=df['sore_throat'].astype(int)
df['shortness_of_breath']=df['shortness_of_breath'].astype(int)
df['head_ache']=df['head_ache'].astype(int)
df.drop(df[(df['corona_result'] =="positive") & (df['cough'] ==1) & (df['fever'] ==0) & (df['sore_throat'] ==0) & (df['shortness_of_breath'] ==0) & (df['head_ache'] == 0  )].index, inplace = True)
df.drop(df[(df['corona_result'] =="positive") & (df['cough'] ==0) & (df['fever'] ==1) & (df['sore_throat'] ==0) & (df['shortness_of_breath'] ==0) & (df['head_ache'] == 0  )].index, inplace = True)
df.drop(df[(df['corona_result'] =="positive") & (df['cough'] ==0) & (df['fever'] ==0) & (df['sore_throat'] ==1) & (df['shortness_of_breath'] ==0) & (df['head_ache'] == 0  )].index, inplace = True)
df.drop(df[(df['corona_result'] =="positive") & (df['cough'] ==0) & (df['fever'] ==0) & (df['sore_throat'] ==0) & (df['shortness_of_breath'] ==1) & (df['head_ache'] == 0  )].index, inplace = True)
df.drop(df[(df['corona_result'] =="positive") & (df['cough'] ==0) & (df['fever'] ==0) & (df['sore_throat'] ==0) & (df['shortness_of_breath'] ==0) & (df['head_ache'] == 1 )].index, inplace = False)
catcols=list(df.select_dtypes(include="object").columns)
xdfohe=pd.get_dummies(df,columns=catcols,drop_first=True)
xdfohe=xdfohe.rename({'test_indication_Contact with confirmed':'test_indication_Contact_with_confirmed'},axis=1)
X=xdfohe.drop({"corona_result_positive"}, axis=1)
Y=xdfohe["corona_result_positive"]
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import NearMiss
smk = SMOTETomek(random_state=42)
X_res,y_res=smk.fit_resample(X,Y)
Xtrain,Xtest,ytrain,ytest=model_selection.train_test_split(X_res,y_res,test_size=.3,random_state=42)
rfeobj=feature_selection.RFE(ensemble.RandomForestClassifier(random_state=42),n_features_to_select=7)
rfeobj.fit(Xtrain,ytrain)
impcols=Xtrain.columns[rfeobj.support_]
Xtrain=Xtrain[impcols]
rf=ensemble.RandomForestClassifier(random_state=42,n_estimators=100,oob_score=True) 
rf.fit(Xtrain,ytrain)
pickle.dump(rf,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))



