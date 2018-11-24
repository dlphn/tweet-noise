from classification2 import Classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

current_file = "C:\\Users\\Public\\Documents\\tweets_2018-11-23T091721.577598.csv"
#dataframe = pd.read_csv(current_file, encoding="utf-8")
#print(dataframe.head())
#id,nb_follower,nb_following,verified,reputation,age,nb_tweets,time,proportion_spamwords,
# orthographe,nb_emoji,RT,spam)
columns = dataframe.columns.values.tolist()

def show_features():
    y = dataframe.iloc[:,-1]
    j=0
    for i in [1,2,4,5] :
        X = dataframe.iloc[:, i]
        j += 1
        plt.subplot(2,2,j)
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    j = 0
    for i in [6,7] :
        j += 1
        X = dataframe.iloc[:, i]
        plt.subplot(2,1,j)
        if i == 6 :
            plt.axis(xmax = 100000)
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    i=0
    for n in [3,8,9,10] :
        i +=1
        x = dataframe.iloc[:, n]
        c = Counter(zip(x, y))
        a = Counter ( y )
        s = [ 30*c[(xx, yy)]/a [(yy)] for xx, yy in zip(x, y)]
        plt.subplot(2,2,i)
        plt.scatter(x, y, s=s)
        plt.xlabel('Paramètre {}'.format(columns[n]))
        plt.ylabel('Spam')
    plt.show()

    i = 0
    for n in [11,12,13] :
        i +=1
        x = dataframe.iloc[:, n]
        c = Counter(zip(x, y))
        a = Counter ( y )
        s = [ 30*c[(xx, yy)]/a [(yy)] for xx, yy in zip(x, y)]
        plt.subplot(2,2,i)
        plt.scatter(x, y, s=s)
        plt.xlabel('Paramètre {}'.format(columns[n]))
        plt.ylabel('Spam')
    plt.show()

    i = 0
    for n in [11,12]:
        i += 1
        x = dataframe.iloc[:, n]
        c = Counter(zip(x, y))
        a = Counter(y)
        s = [30 * c[(xx, yy)] / a[(yy)] for xx, yy in zip(x, y)]
        plt.subplot(2, 2, i)
        plt.scatter(x, y, s=s)
        plt.xlabel('Paramètre {}'.format(columns[n]))
        plt.ylabel('Spam')
    plt.show()


def PCA(X,y):
    #sc = StandardScaler()
    #Z = sc.fit_transform(X)
    #print(Z)
    pca = decomposition.PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2','3'])
    #print(principalDf)
    result = pd.concat([principalDf, y], axis=1)
    print(pca.explained_variance_ratio_)
    return result

def correlation_matrix(dataset):
    columns = dataset.columns.values.tolist()
    correlations = dataset.corr()
    #print(correlations)
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, len(columns), 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(columns)
    ax.set_yticklabels(columns)
    plt.show()

def affichage(result):
    mask = result.iloc[:,-1] == 0
    negative = plt.scatter(result.iloc[:,0][~mask].values, result.iloc[:,1][~mask].values)
    positive = plt.scatter(result.iloc[:,0][mask].values,result.iloc[:,1][mask].values)
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    plt.legend((positive, negative), ('actu', 'non actu'))
    plt.show()

classif = Classification()
dataframe = classif.create_dataframe()
columns = dataframe.columns.values.tolist()
#X = pd.concat([dataframe.iloc[:,1:7], dataframe.iloc[:,8:-1]], axis=1)
X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]
df2 = dataframe[['nb_follower', 'nb_following', 'verified', 'nb_tweets', 'proportion_spamwords', 'proportion_whitewords',  'orthographe',  'nb_emoji']]
#print(X.head())
#print(correlation_matrix(X))
a = PCA(df2,y)
affichage(a)

#show_features()