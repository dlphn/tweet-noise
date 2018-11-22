from classification2 import Classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler

current_file = "D:\\valentin\\documents\\Centrale\\3A\\OSY\\TweetNoise\\tweets_2018-11-22T142836.815852.csv"
dataframe = pd.read_csv(current_file, encoding="utf-8")
#print(dataframe.head())
#id,nb_follower,nb_following,verified,reputation,age,nb_tweets,time,proportion_spamwords,
# orthographe,nb_emoji,RT,spam)
columns = dataframe.columns.values.tolist()
print(columns)

def show_features():
    y = dataframe.iloc[:,-1]
    for i in [1,2,4,5] :
        X = dataframe.iloc[:, i]
        if i == 5 :
            j = 3
        else :
            j =i
        plt.subplot(2,2,j)
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    for i in [6,9] :
        X = dataframe.iloc[:, i]
        plt.subplot(2,2,i-5)
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    i=0
    for n in [3,7,8,10] :
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
    sc = StandardScaler()
    #Z = sc.fit_transform(X)
    #print(Z)
    pca = decomposition.PCA()
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2','3','4','5','6','7','8','9','10','11'])
    #print(principalDf)
    result = pd.concat([principalDf, y], axis=1)
    print(pca.explained_variance_ratio_)
    #return result

def correlation_matrix(dataset):
    HEADERS = ['nb_follower','nb_following','verified','reputation','age','nb_tweets','proportion_spamwords',
               'proportion_whitewords','orthographe','nb_hashtag','guillements','nb_emoji','spam']
    correlations = dataset.corr()
    #print(correlations)
    # plot correlation matrix
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0, 10, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(HEADERS)
    ax.set_yticklabels(HEADERS)
    plt.show()

def affichage (result):
    mask = result.iloc[:,-1] == 0
    negative = plt.scatter(result.iloc[:,0][~mask].values, result.iloc[:,1][~mask].values)
    positive = plt.scatter(result.iloc[:,0][mask].values,result.iloc[:,1][mask].values)
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    plt.legend((positive, negative), ('actu', 'non actu'))
    #plt.show()

#classif = Classification()
#dataframe = classif.create_dataframe()
#X = pd.concat([dataframe.iloc[:,1:7], dataframe.iloc[:,8:-1]], axis=1)
X = dataframe.iloc[:,:-1]
y = dataframe.iloc[:,-1]
#print(X.head())
#print(correlation_matrix(X))
#print(PCA(X,y))

show_features()