#https://www.google.com/url?q=https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60&sa=D&source=calendar&ust=1618944053033000&usg=AOvVaw0NQFCvsyHJ_E1zI6_kb__w
import pandas as pd

url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#load dataset into Pandas DataFrame
df = pd.read_csv(url ,  names = ['sepal length' , 'sepal width', 'petal length' , 'petal width', 'target'])
# print(df.to_string())

from sklearn.preprocessing import StandardScaler

features = ['sepal length' , 'sepal width' , 'petal length' , 'petal width']

#separating out the features
x = df.loc[:, features].values

#separating out the target
y = df.loc[:,['target']].values

#Standardizing the features
x = StandardScaler().fit_transform(x)
# print(features)
# print(x)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
finalDF = pd.concat([principalDf, df[['target']]], axis=1)
# print(principalDf)
print(finalDF)