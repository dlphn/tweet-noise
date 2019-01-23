# from classification2 import Classification
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
import seaborn as sb

from config import FILEDIR

# Data frame original
current_file = FILEDIR + "tweets_2019-01-23T14:26:33.598574.csv"
dataframe = pd.read_csv(current_file, encoding="utf-8")


# print(dataframe.head())


# Normalisation du df
def categorize_bool(x):
    if x:
        return 1
    else:
        return 0


# print(dataframe.verified)
dataframe.spam = dataframe.spam.apply(categorize_bool)
df2 = pd.concat([dataframe.iloc[:, 1:7], dataframe.iloc[:, 8:]], axis=1)
# print(dataframe.columns.tolist())
# print(dataframe.describe())

"""for i in range (5):
    df2.iloc[:,i] = ((df2.iloc[:,i] - df2.iloc[:,i].mean()) / (df2.iloc[:,i].max() - df2.iloc[:,i].min()))
sb.pairplot(df2.iloc[:,:i])
plt.show()"""

plt.figure(figsize=(10, 10))
for column_index, column in enumerate(df2.columns):
    if column == 'spam':
        continue
    plt.subplot(4, 4, column_index + 1)
    sb.violinplot(x='spam', y=column, data=df2)
plt.show()


# Data frame classifié
# classif = Classification()
# dataframe = classif.create_dataframe()

def show_features(dataset):
    columns = dataset.columns.values.tolist()
    y = dataset.iloc[:, -1]
    j = 0
    for i in [1, 2, 4, 5]:
        j += 1
        plt.subplot(2, 2, j)
        if j == 1:
            plt.axis(xmin=0, xmax=500000)
        if j == 2:
            plt.axis(xmax=4000)
        X = dataframe.iloc[:, i]
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    j = 0
    for i in [6, 7]:
        j += 1
        X = dataframe.iloc[:, i]
        plt.subplot(2, 1, j)
        if i == 6:
            plt.axis(xmax=100000)
        plt.scatter(X, y)
        plt.xlabel('Paramètre {}'.format(columns[i]))
        plt.ylabel('Spam')
    plt.show()

    i = 0
    for n in [3, 8, 9, 10]:
        i += 1
        x = dataframe.iloc[:, n]
        c = Counter(zip(x, y))
        a = Counter(y)
        s = [60 * c[(xx, yy)] / a[(yy)] for xx, yy in zip(x, y)]
        plt.subplot(2, 2, i)
        plt.scatter(x, y, s=s)
        plt.xlabel('Paramètre {}'.format(columns[n]))
        plt.ylabel('Spam')
    plt.show()

    i = 0
    for n in [11, 12, 13]:
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

    i = 0
    for n in [11, 12]:
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


def PCA(X, y):
    # sc = StandardScaler()
    # Z = sc.fit_transform(X)
    # print(Z)
    pca = decomposition.PCA(n_components=3)
    principalComponents = pca.fit_transform(X)
    principalDf = pd.DataFrame(data=principalComponents
                               , columns=['principal component 1', 'principal component 2', '3'])
    # print(principalDf)
    result = pd.concat([principalDf, y], axis=1)
    print(pca.explained_variance_ratio_)
    return result


def correlation_matrix(dataset):
    columns = dataset.columns.values.tolist()
    correlations = dataset.corr()
    # print(columns)
    # print(correlations)
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
    mask = result.iloc[:, -1] == 0
    negative = plt.scatter(result.iloc[:, 0][~mask].values, result.iloc[:, 1][~mask].values)
    positive = plt.scatter(result.iloc[:, 0][mask].values, result.iloc[:, 1][mask].values)
    plt.xlabel('pca 1')
    plt.ylabel('pca 2')
    plt.legend((positive, negative), ('actu', 'non actu'))
    plt.show()


def plot_features(dataframe, i, j):
    columns_name = dataframe.columns.values.tolist()
    x = dataframe.iloc[:, i]
    y = dataframe.iloc[:, j]
    c = Counter(zip(x, y))
    a = Counter(y)
    s = [30 * c[(xx, yy)] / a[(yy)] for xx, yy in zip(x, y)]
    plt.scatter(x, y, s=s)
    plt.xlabel(columns_name[i])
    plt.ylabel(columns_name[j])
    plt.show()


def visualisation_multi(dataframe):
    desag_finale = dataframe.columns.tolist()
    classes = list(map(int, desag_finale[:, -1]))
    nbre_classes = len(set(classes))
    cmap = plt.get_cmap('rainbow')

    #    t, data_p = import_data_prise(data_path_p)

    classes = list(map(int, desag_finale[:, -1]))
    count = 0

    arguments = [1, 3, 4, 5, 6, 7, 8]
    label = ['Puissance', 'Durée', 'Variance', 'Variance inf', 'Variance sup', 'Skewness', 'Kurtosis']
    plt.figure(1, tight_layout=True)
    for x in range(7):
        for y in range(7):
            count += 1
            for i in set(classes):
                col = cmap(i / nbre_classes)
                pp_k = np.where(desag_finale[:, -1] == i)[0]  # indice dans S_f pour le cluster examiné
                plt.subplot(7, 7, x + (y) * 7 + 1)
                plt.scatter(desag_finale[pp_k, arguments[x]], desag_finale[pp_k, arguments[y]], color=col)


X = pd.concat([dataframe.iloc[:, 3:7], dataframe.iloc[:, 8:-1]], axis=1)
# X = dataframe.iloc[:,1:-1]
y = dataframe.iloc[:, -1]
# print(X.head())

# df2 = dataframe[['nb_follower', 'nb_following', 'verified', 'nb_tweets', 'proportion_spamwords', 'proportion_whitewords',  'orthographe',  'nb_emoji']]
# print(plot_features(df2,0,1))

# print(correlation_matrix(X))

# a = PCA(df2,y)
# affichage(a)

# show_features(dataframe)
