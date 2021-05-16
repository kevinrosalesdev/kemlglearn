from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from kemlglearn.cluster.MinMaxKMeans import MinMaxKMeans
from kemlglearn.datasets import make_blobs


def test_generated_dataset():
    X, y_data = make_blobs(n_samples=1000, n_features=2, centers=7, random_state=0)

    plt.scatter(X[:, 0], X[:, 1], c=y_data)
    plt.title("Generated Dataset Ground Truth")
    plt.show()

    MMKmeans = MinMaxKMeans(n_clusters=7, random_state=seed)
    predict = MMKmeans.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=predict)
    plt.title("Generated Dataset MinMax k-Means Prediction")
    plt.show()

    k_means = KMeans(n_clusters=7, init='random', random_state=seed)
    predict = k_means.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=predict)
    plt.title("Generated Dataset k-Means Prediction")
    plt.show()


def test_iris_dataset():
    X, y_data = load_iris(return_X_y=True)

    plt.scatter(X[:, 1], X[:, 3], c=y_data)
    plt.title("Iris Dataset Ground Truth")
    plt.show()

    MMKmeans = MinMaxKMeans(n_clusters=3, random_state=seed)
    predict = MMKmeans.fit_predict(X)

    plt.scatter(X[:, 1], X[:, 3], c=predict)
    plt.title("Iris Dataset MinMax k-Means Prediction")
    plt.show()

    k_means = KMeans(n_clusters=3, init='random', random_state=seed)
    predict = k_means.fit_predict(X)

    plt.scatter(X[:, 1], X[:, 3], c=predict)
    plt.title("Generated Dataset k-Means Prediction")
    plt.show()


if __name__ == '__main__':
    seed = 0

    test_generated_dataset()
    test_iris_dataset()
