import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from pylab import rcParams
import pandas as pd
import seaborn as sns
from sklearn import  tree
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
def plot_correlation(data):
    rcParams['figure.figsize'] = 15, 25
    fig = plt.figure()
    sns.heatmap(data.corr(), annot=True, fmt=".1f")
    fig.savefig('correlation_coefficient.png')



df = pd.read_csv('heart.csv')


#we check the names of the columns on the on name file.
df.columns = ["age","sex","BP","cholestrol","heart disease"]

plot_correlation(df)

df_x = df[["age","sex","BP","cholestrol"]]
df_y = df[["heart disease"]]

X_train, X_test, y_train, y_test = train_test_split(df_x,
                                                    df_y, test_size = 0.2,
                                                    random_state = 0)

clf = tree.DecisionTreeClassifier(random_state=0,max_depth=8)

clf = clf.fit(X_train, y_train)
             
# Make predictions using the testing set
y_pred = clf.predict(X_test)


accuracy_score(y_test, y_pred)

df.corr()

age=input('write your age: ')
sex=input('write your sex(female=0,male=1): ')
BP=input('write your blood pressure: ')
cholestrol=input('write your cholestrol: ')


numpy_array = np.array([[age,sex,BP,cholestrol]])
new_df = pd.DataFrame(numpy_array, columns=['age','sex','BP','cholestrol'])

pred = clf.predict(new_df)
if(pred[0]==1):
    print('You should visit the doctor')
else:
     print('Everything is okay')


