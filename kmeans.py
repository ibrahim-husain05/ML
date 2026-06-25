from sklearn.cluster import KMeans
from sklearn.datasets import load_iris  
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
# print(df.head())
df['flower']=iris.target
df.drop(['sepal length (cm)','sepal width (cm)','flower'],axis=1,inplace=True)
# print(df.head())

kmeans=KMeans(n_clusters=3)
kmeans.fit(df)
yp=kmeans.predict(df)
# print(yp)
df['cluster']=yp
print(df.head())
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==2]
plt.scatter(df1['petal length (cm)'],df1['petal width (cm)'],color='green')
plt.scatter(df2['petal length (cm)'],df2['petal width (cm)'],color='red')
plt.scatter(df3['petal length (cm)'],df3['petal width (cm)'],color='blue')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('K-Means Clustering')
# plt.show()  
Sse=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df)
    Sse.append(kmeans.inertia_)
plt.xlabel('i')
# sum of sqared error sse
plt.ylabel('SSE')
plt.title('Elbow Method')
plt.plot(range(1,11),Sse)
plt.show()
