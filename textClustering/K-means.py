from sklearn import preprocessing
import pandas as pd
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import decomposition

dataframe = pd.read_csv('C:\\Users\\Public\\Documents\\new.csv', sep = ',', header = 0)
del dataframe["Unnamed: 3"]
#print(dataframe.describe())

#Visualisation des Clusters
def Draw(pred, dataframe, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii,pp in enumerate(pred):
        plt.scatter(dataframe.iloc[:,2][ii],dataframe.iloc[:,3][ii],color = colors[pred[ii]], s = 16)
        if dataframe.iloc[:,0][ii] == False:
            plt.scatter(dataframe.iloc[:,2][ii],dataframe.iloc[:,3][ii], color="y", marker="*", s= 20)
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    #plt.savefig(name)
    plt.show()

#Remise en forme du Dataframe, une ligne = un tweet
def shape_df (dataframe):
    dataframe.vectorize = dataframe.vectorize.map(lambda x: x.lstrip('[').rstrip(']'))
    temp = dataframe.vectorize.str.split( expand=True)
    temp.columns = ['vect_{}'.format(n) for n in range(1, len(temp.columns) + 1)]
    dataframe = dataframe.drop('vectorize', axis=1).join(temp)
    #print(dataframe.describe())
    return dataframe

#Reduction de dimensionalité pour facilité les calculs
def PCA(X,y,n):
    pca = decomposition.PCA(n_components=n)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component {}'.format(i) for i in range(1,n+1) ])
    #print(principalDf)
    result = pd.concat([y,principalDf], axis=1)
    #print(pca.explained_variance_ratio_)
    return result

#Clusterization par les méthodes des K-means pour n clusters
def Clusterize(df,n):
    clusterize = KMeans(n_clusters=n, n_init = 50)
    clusterize.fit(df.iloc[:, 2:9])
    pred = clusterize.predict(df_pca.iloc[:, 2:9])
    return clusterize.inertia_, pred

#fonction pour visualiser l'erreur en fonction du nombre de Cluster
def choose_K(n):
    cost_list = []
    for i in range(2,n+1):
        cost, pred = Clusterize(df_pca,i)
        cost_list.append(cost)
    print(cost_list)
    plt.plot([i for i in range(2,len(cost_list)+2)],cost_list)
    plt.show()

dataframe = shape_df(dataframe)
df_pca = PCA(dataframe.iloc[:,2:30],dataframe.iloc[:,1],8)
inertia, pred = Clusterize(df_pca,5)

try:
    Draw(pred, df_pca, name="clusters.pdf", f1_name='pca 1', f2_name='pca 2')
except NameError:
    print ("no predictions object named pred found, no clusters to plot")

