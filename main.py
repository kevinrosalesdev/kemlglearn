import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score

from kemlglearn.cluster.MinMaxKMeans import MinMaxKMeans, ssd
from kemlglearn.feature_selection.unsupervised import LaplacianScore
from kemlglearn.preprocessing import datasets


def get_e(data, labels, clusterizer):
    return np.array([np.sum(ssd(data[np.where(labels == k)],
                                [clusterizer.cluster_centers_[k]]).reshape(-1))
                    for k in range(clusterizer.n_clusters)])


def test_dataset(data, labels, name):
    """
    ls = LaplacianScore(n_neighbors=10, bandwidth=0.05)
    ls.fit(data)
    best = ls._best_k_scores(2)
    """

    pca = PCA(n_components=2)
    x = pca.fit_transform(data)

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.title(f"[{name}] Ground Truth")
    plt.show()

    MMKmeans = MinMaxKMeans(n_clusters=np.unique(labels).shape[0], beta=0.3, random_state=seed)
    predict = MMKmeans.fit_predict(data)
    print(f"[{name}] MinMax k-Means E_max:", np.max(get_e(data, predict, MMKmeans)))
    print(f"[{name}] MinMax k-Means E_sum:", np.sum(get_e(data, predict, MMKmeans)))
    print(f"[{name}] NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means Prediction")
    plt.show()

    k_means = KMeans(n_clusters=np.unique(labels).shape[0], init='random', random_state=seed)
    predict = k_means.fit_predict(data)
    print(f"[{name}] k-Means E_max:", np.max(get_e(data, predict, k_means)))
    print(f"[{name}] k-Means E_sum:", np.sum(get_e(data, predict, k_means)))
    print(f"[{name}] NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] k-Means Prediction")
    plt.show()

    print()


if __name__ == '__main__':
    seed = 0

    x, y = datasets.load_generated_dataset()
    test_dataset(data=x, labels=y, name='Generated Dataset')
    x, y = datasets.load_iris_dataset()
    test_dataset(data=x, labels=y, name='Iris')
    x, y = datasets.load_car_evaluation_dataset()
    test_dataset(data=x, labels=y, name='Car Evaluation')
    x, y = datasets.load_pendigits_dataset()
    test_dataset(data=x, labels=y, name='Pendigits')
    x, y = datasets.load_ecoli_dataset()
    test_dataset(data=x, labels=y, name='Ecoli')
    x, y = datasets.load_dermatology_dataset()
    test_dataset(data=x, labels=y, name='Dermatology')
