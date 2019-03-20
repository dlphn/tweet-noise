from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
from subprocess import call
from IPython.display import Image
from classification.randomForest import *

HEADERS = dataset.columns.values.tolist()
feature_headers = ['nb_follower', 'nb_following', 'verified', 'reputation', 'age', 'nb_tweets', 'posted_at', 'length',
                   'proportion_spamwords', 'orthographe', 'nb_hashtag', 'nb_urls', 'nb_emoji']
class_names = ["spam", "actualité"]

def random_forest_visualization (HEADERS, feature_headers, class_names) :
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.7, ['nb_follower', 'nb_following', 'verified', 'reputation',
                                                                'age', 'nb_tweets', 'posted_at', 'length',
                                                                'proportion_spamwords', 'orthographe', 'nb_hashtag',
                                                                'nb_urls', 'nb_emoji', 'type'], HEADERS[-2])
    model = RandomForestClassifier(class_weight= {0: 1, 1: 5}, max_depth= 10, min_samples_leaf= 5, min_samples_split= 20)
    model.fit(train_x.drop('type',axis=1), train_y)
    estimator = model.estimators_[9]
    return (export_graphviz(estimator, out_file='tree.dot',
                feature_names = feature_headers,
                class_names = class_names,
                rounded = True, proportion = False,
                precision = 2, filled = True))

if __name__ == "__main__":
    random_forest_visualization (HEADERS, feature_headers, class_names)
    call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
    Image(filename = 'tree.png')

