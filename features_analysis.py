from classification2 import Classification
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

current_file = "C:\\Users\\Public\\Documents\\tweets_2018-11-05T22_47_26.114536.csv"
dataframe = pd.read_csv(current_file, encoding="utf-8")
#print(dataframe.head())
columns = dataframe.columns.values.tolist()
#print(columns)


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

for i in [6,7,8,9] :
    X = dataframe.iloc[:, i]
    plt.subplot(2,2,i-5)
    plt.scatter(X, y)
    plt.xlabel('Paramètre {}'.format(columns[i]))
    plt.ylabel('Spam')
plt.show()

i=0
for n in [3,10,11] :
    i +=1
    x = dataframe.iloc[:, n]
    c = Counter(zip(x, y))
    s = [ 0.5*c[(xx, yy)] for xx, yy in zip(x, y)]
    plt.subplot(2,2,i)
    plt.scatter(x, y, s=s)
    plt.xlabel('Paramètre {}'.format(columns[n]))
    plt.ylabel('Spam')
plt.show()


