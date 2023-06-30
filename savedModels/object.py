import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from packageC45.c45 import TreeNode
df=pd.read_csv("Data_Set.csv")
X=pd.DataFrame(df.iloc[:,1:7])
y=df.iloc[:,7:8]
X.columns=['la','Lon','ecp','mssw','pd','sst']

from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
d1=le.fit_transform(y)
y=pd.Series(d1)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

tree_cyclone = TreeNode(min_samples_split=3, max_depth=4, seed=2)
#Training the model
tree_cyclone.fit(X_train, y_train)

prediction = X_test.apply(lambda row : tree_cyclone.predict(row), axis = 1)
print(accuracy_score(y_test,prediction))

pred=tree_cyclone.predict({'la':10.4,'lon':52,'ecp':982,'mssw':65,'pd':22,'sst':26.1})
print(pred)

import pickle

with open('model.pkl', 'wb') as f:
    pickle.dump(tree_cyclone, f)