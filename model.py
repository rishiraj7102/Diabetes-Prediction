import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

import warnings
warnings.filterwarnings('ignore')
import pickle
df=pd.read_csv("kaggle_diabetes.csv")
X=df.iloc[:,:8]
Y=df.iloc[:,8:]
Y=Y.values

randomforest=RandomForestClassifier()
randomforest.fit(X,Y)
cross_val_score(randomforest,X,Y,cv=5).mean()

filename="diabetes-prediction-rfc-model.pkl"
pickle.dump(randomforest,open(filename,'wb'))