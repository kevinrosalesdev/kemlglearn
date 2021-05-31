import warnings
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, calinski_harabasz_score, silhouette_score, davies_bouldin_score
from time import time

from kemlglearn.cluster.MinMaxKMeans import MinMaxKMeans, ssd
from kemlglearn.feature_selection.unsupervised import LaplacianScore
from kemlglearn.preprocessing import datasets

warnings.filterwarnings("ignore")


def get_e(data, labels, clusterizer):
    return np.array([np.sum(ssd(data[np.where(labels == k)],
                                [clusterizer.cluster_centers_[k]]).reshape(-1))
                     if np.where(labels == k)[0].shape[0] > 0 else 0
                     for k in range(clusterizer.n_clusters)])


def get_k_metrics(data, name, max_clusters=6):
    scores = []
    for nc in range(2, max_clusters+1):
        MMKmeans = MinMaxKMeans(n_clusters=nc, beta=0.1, random_state=0)
        predict = MMKmeans.fit_predict(data)
        scores.append((silhouette_score(data, predict),
                      calinski_harabasz_score(data, predict),
                      davies_bouldin_score(data, predict)))

    fig = plt.figure(figsize=(15, 5))
    plt.suptitle(f"{name} Dataset Internal Criteria")
    for idx, title in enumerate(['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin'], 1):
        fig.add_subplot(1, 3, idx)
        plt.title(title)
        plt.ylabel('Score')
        plt.xlabel('K')
        plt.grid()
        plt.plot(range(2, max_clusters+1), [score[idx-1] for score in scores], '-o')
    plt.tight_layout()
    plt.show()


def get_table(data, labels, name, n_iter=50):
    df = pd.DataFrame(columns=['Method', 'E_max', 'E_sum', 'NMI', 'Time'])

    for beta in [0, 0.1, 0.3]:
        print(f"Beta: {beta}")
        e_max_list = []
        e_sum_list = []
        nmi_list = []
        time_list = []
        e_max_list_km = []
        e_sum_list_km = []
        nmi_list_km = []
        time_list_km = []
        for i in range(n_iter):
            print(f"Iter: {i+1}/{n_iter}")
            MMKmeans = MinMaxKMeans(n_clusters=np.unique(labels).shape[0], beta=beta, random_state=i)
            start_time = time()
            predict = MMKmeans.fit_predict(data)
            mmkmeans_time = time() - start_time
            time_list.append(mmkmeans_time)
            e = get_e(data, predict, MMKmeans)
            e_max_list.append(np.max(e))
            e_sum_list.append(np.sum(e))
            nmi_list.append(normalized_mutual_info_score(labels, predict))

            k_means = KMeans(n_clusters=np.unique(labels).shape[0], init=np.array(MMKmeans.cluster_centers_),
                             n_init=1, random_state=i, max_iter=500, tol=10E-6, n_jobs=1)
            start_time = time()
            predict = k_means.fit_predict(data)
            time_list_km.append(mmkmeans_time + (time() - start_time))
            e = get_e(data, predict, k_means)
            e_max_list_km.append(np.max(e))
            e_sum_list_km.append(np.sum(e))
            nmi_list_km.append(normalized_mutual_info_score(labels, predict))

        df = df.append({'Method': f'MinMax (beta={beta})',
                        'E_max': f'{round(np.mean(e_max_list), 2)} ± {round(np.std(e_max_list), 2)}',
                        'E_sum': f'{round(np.mean(e_sum_list), 2)} ± {round(np.std(e_sum_list), 2)}',
                        'NMI': f'{round(np.mean(nmi_list), 2)} ± {round(np.std(nmi_list), 2)}',
                        'Time': f'{round(np.mean(time_list), 2)} ± {round(np.std(time_list), 2)}'},
                       ignore_index=True)

        df = df.append({'Method': f'MinMax (beta={beta}) + k-Means',
                        'E_max': f'{round(np.mean(e_max_list_km), 2)} ± {round(np.std(e_max_list_km), 2)}',
                        'E_sum': f'{round(np.mean(e_sum_list_km), 2)} ± {round(np.std(e_sum_list_km), 2)}',
                        'NMI': f'{round(np.mean(nmi_list_km), 2)} ± {round(np.std(nmi_list_km), 2)}',
                        'Time': f'{round(np.mean(time_list_km), 2)} ± {round(np.std(time_list_km), 2)}'},
                       ignore_index=True)

    e_max_list = []
    e_sum_list = []
    nmi_list = []
    time_list = []
    for i in range(n_iter):
        print(f"Iter: {i+1}/{n_iter}")
        k_means = KMeans(n_clusters=np.unique(labels).shape[0],
                         init='random', n_init=1, random_state=i, max_iter=500, tol=10E-6, n_jobs=1)
        start_time = time()
        predict = k_means.fit_predict(data)
        time_list.append(time() - start_time)
        e = get_e(data, predict, k_means)
        e_max_list.append(np.max(e))
        e_sum_list.append(np.sum(e))
        nmi_list.append(normalized_mutual_info_score(labels, predict))

    df = df.append({'Method': f'k-Means',
                    'E_max': f'{round(np.mean(e_max_list), 2)} ± {round(np.std(e_max_list), 2)}',
                    'E_sum': f'{round(np.mean(e_sum_list), 2)} ± {round(np.std(e_sum_list), 2)}',
                    'NMI': f'{round(np.mean(nmi_list), 2)} ± {round(np.std(nmi_list), 2)}',
                    'Time': f'{round(np.mean(time_list), 2)} ± {round(np.std(time_list), 2)}'},
                   ignore_index=True)

    e_max_list = []
    e_sum_list = []
    nmi_list = []
    time_list = []
    for i in range(n_iter):
        print(f"Iter: {i+1}/{n_iter}")
        k_means = KMeans(n_clusters=np.unique(labels).shape[0],
                         init='k-means++', n_init=1, random_state=i, max_iter=500, tol=10E-6, n_jobs=1)
        start_time = time()
        predict = k_means.fit_predict(data)
        time_list.append(time() - start_time)
        e = get_e(data, predict, k_means)
        e_max_list.append(np.max(e))
        e_sum_list.append(np.sum(e))
        nmi_list.append(normalized_mutual_info_score(labels, predict))

    df = df.append({'Method': f'k-Means++',
                    'E_max': f'{round(np.mean(e_max_list), 2)} ± {round(np.std(e_max_list), 2)}',
                    'E_sum': f'{round(np.mean(e_sum_list), 2)} ± {round(np.std(e_sum_list), 2)}',
                    'NMI': f'{round(np.mean(nmi_list), 2)} ± {round(np.std(nmi_list), 2)}',
                    'Time': f'{round(np.mean(time_list), 2)} ± {round(np.std(time_list), 2)}'},
                   ignore_index=True)


    agg_clust = AgglomerativeClustering(n_clusters=np.unique(labels).shape[0])
    start_time = time()
    predict = agg_clust.fit_predict(data)
    final_time = time() - start_time
    nmi = normalized_mutual_info_score(labels, predict)

    df = df.append({'Method': f'Agglomerative Clustering',
                    'E_max': f'-',
                    'E_sum': f'-',
                    'NMI': f'{round(nmi, 2)}',
                    'Time': f'{round(final_time, 2)}'},
                   ignore_index=True)

    df.to_csv(f'Out/evaluation/{name}.csv', index=False)


def test_dataset(data, labels, name):

    # ls = LaplacianScore(n_neighbors=10, bandwidth=0.05)
    # ls.fit(data)
    # best = ls._best_k_scores(2)

    pca = PCA(n_components=2)
    x = pca.fit_transform(data)

    plt.scatter(x[:, 0], x[:, 1], c=labels)
    plt.title(f"[{name}] Ground Truth")
    plt.savefig(f"Out/plots/{name}-GT")
    plt.close()

    MMKmeans = MinMaxKMeans(n_clusters=np.unique(labels).shape[0], beta=0, random_state=0)
    predict = MMKmeans.fit_predict(data)
    # print(f"[{name}] MinMax k-Means (beta = 0) E_max:", np.max(get_e(data, predict, MMKmeans)))
    # print(f"[{name}] MinMax k-Means (beta = 0) E_sum:", np.sum(get_e(data, predict, MMKmeans)))
    # print(f"[{name}] MinMax k-Means (beta = 0) NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=0")
    plt.close()

    k_means = KMeans(n_clusters=np.unique(labels).shape[0], init=np.array(MMKmeans.cluster_centers_),
                     n_init=1, random_state=0, max_iter=500, tol=10E-6, n_jobs=1)
    predict = k_means.fit_predict(data)
    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means + k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=0-KMeans")
    plt.close()

    MMKmeans = MinMaxKMeans(n_clusters=np.unique(labels).shape[0], beta=0.1, random_state=0)
    predict = MMKmeans.fit_predict(data)

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=01")
    plt.close()

    k_means = KMeans(n_clusters=np.unique(labels).shape[0], init=np.array(MMKmeans.cluster_centers_),
                     n_init=1, random_state=0, max_iter=500, tol=10E-6, n_jobs=1)
    predict = k_means.fit_predict(data)
    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means + k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=01-KMeans")
    plt.close()

    MMKmeans = MinMaxKMeans(n_clusters=np.unique(labels).shape[0], beta=0.3, random_state=0)
    predict = MMKmeans.fit_predict(data)

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=03")
    plt.close()

    k_means = KMeans(n_clusters=np.unique(labels).shape[0], init=np.array(MMKmeans.cluster_centers_),
                     n_init=1, random_state=0, max_iter=500, tol=10E-6, n_jobs=1)
    predict = k_means.fit_predict(data)
    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] MinMax k-Means + k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-MMKMeans-beta=03-KMeans")
    plt.close()

    k_means = KMeans(n_clusters=np.unique(labels).shape[0],
                     init='random', n_init=1, random_state=0, max_iter=500, tol=10E-6, n_jobs=1)
    predict = k_means.fit_predict(data)
    # print(f"[{name}] k-Means E_max:", np.max(get_e(data, predict, k_means)))
    # print(f"[{name}] k-Means E_sum:", np.sum(get_e(data, predict, k_means)))
    # print(f"[{name}] k-Means NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] k-Means Prediction")
    plt.savefig(f"Out/plots/{name}-KMeans")
    plt.close()

    k_means_pp = KMeans(n_clusters=np.unique(labels).shape[0],
                        init='k-means++', n_init=1, random_state=0, max_iter=500, tol=10E-6, n_jobs=1)
    predict = k_means_pp.fit_predict(data)
    # print(f"[{name}] k-Means++ E_max:", np.max(get_e(data, predict, k_means_pp)))
    # print(f"[{name}] k-Means++ E_sum:", np.sum(get_e(data, predict, k_means_pp)))
    # print(f"[{name}] k-Means++ NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] k-Means++ Prediction")
    plt.savefig(f"Out/plots/{name}-KMeansPP")
    plt.close()

    agg_clust = AgglomerativeClustering(n_clusters=np.unique(labels).shape[0])
    predict = agg_clust.fit_predict(data)
    # print(f"[{name}] Agglomerative Clustering NMI: {normalized_mutual_info_score(labels, predict)}")

    plt.scatter(x[:, 0], x[:, 1], c=predict)
    plt.title(f"[{name}] Agglomerative Clustering Prediction")
    plt.savefig(f"Out/plots/{name}-AC")
    plt.close()


if __name__ == '__main__':

    # x, y = datasets.load_generated_dataset()
    # get_table(data=x, labels=y, name='generated', n_iter=50)
    # x, y = datasets.load_iris_dataset()
    # get_table(data=x, labels=y, name='iris', n_iter=50)
    # x, y = datasets.load_car_evaluation_dataset()
    # get_table(data=x, labels=y, name='car-evaluation', n_iter=50)
    # x, y = datasets.load_pendigits_dataset()
    # get_table(data=x, labels=y, name='pendigits', n_iter=50)
    # x, y = datasets.load_ecoli_dataset()
    # get_table(data=x, labels=y, name='ecoli', n_iter=50)
    # x, y = datasets.load_dermatology_dataset()
    # get_table(data=x, labels=y, name='dermatology', n_iter=50)

    x, y = datasets.load_generated_dataset()
    test_dataset(data=x, labels=y, name='Generated')
    x, y = datasets.load_iris_dataset()
    test_dataset(data=x, labels=y, name='Iris')
    x, y = datasets.load_car_evaluation_dataset()
    test_dataset(data=x, labels=y, name='Car-Evaluation')
    x, y = datasets.load_pendigits_dataset()
    test_dataset(data=x, labels=y, name='Pendigits')
    x, y = datasets.load_ecoli_dataset()
    test_dataset(data=x, labels=y, name='Ecoli')
    x, y = datasets.load_dermatology_dataset()
    test_dataset(data=x, labels=y, name='Dermatology')

    x, y = datasets.load_iris_dataset()
    get_k_metrics(x, name='Iris', max_clusters=10)
