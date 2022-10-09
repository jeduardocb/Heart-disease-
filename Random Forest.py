import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import  tree
df = pd.read_csv('heart.csv')

#we check the names of the columns on the on name file.
df.columns = ["age","sex","BP","cholestrol","heart disease"]


df_x = df[["age","sex","BP","cholestrol"]]
df_y = df[["heart disease"]]


#split data
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y,
                                                    test_size = 0.2,
                                                    random_state = 0)

clf = RandomForestClassifier(max_depth=6, random_state=0)


clf = clf.fit(X_train, y_train.values.ravel())
             
# Make predictions using the testing set
y_pred = clf.predict(X_test)

accuracy_score(y_test, y_pred)

age=input('write your age: ')
sex=input('write your sex(female=0,male=1): ')
BP=input('write your blood pressure: ')
cholestrol=input('write your cholestrol: ')

numpy_array = np.array([[age,sex,BP,cholestrol]])
new_df = pd.DataFrame(numpy_array, columns=['age','sex','BP','cholestrol'])

pred = clf.predict(new_df)
if(pred[0]==1):
    print('OOOPS You should visit the doctor')
else:
     print('Everything is okay')
