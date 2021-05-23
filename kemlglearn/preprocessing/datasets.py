import pandas as pd

from sklearn.datasets import load_iris
from kemlglearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_generated_dataset(n_samples=1000, n_features=2, centers=7, random_state=0):
    x, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers, random_state=random_state)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y


def load_iris_dataset():
    x, y = load_iris(return_X_y=True)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y


def load_car_evaluation_dataset():
    dataset = pd.read_csv('Datasets/car.csv')
    dataset['buying'] = dataset['buying'].map({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    dataset['maint'] = dataset['maint'].map({'low': 0, 'med': 1, 'high': 2, 'vhigh': 3})
    dataset['doors'] = dataset['doors'].map({'2': 0, '3': 1, '4': 2, '5more': 3})
    dataset['persons'] = dataset['persons'].map({'2': 0, '4': 1, 'more': 2})
    dataset['lug_boot'] = dataset['lug_boot'].map({'small': 0, 'med': 1, 'big': 2})
    dataset['safety'] = dataset['safety'].map({'low': 0, 'med': 1, 'high': 2})
    dataset['evaluation'] = dataset['evaluation'].map({'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3})
    dataset.drop(dataset[dataset['evaluation'] == 0].index, inplace=True)
    dataset = dataset.to_numpy()
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y


def load_pendigits_dataset():
    train_dataset = pd.read_csv('Datasets/pendigits-train.csv', header=None)
    test_dataset = pd.read_csv('Datasets/pendigits-test.csv', header=None)
    dataset = pd.concat((train_dataset, test_dataset))
    dataset = dataset.to_numpy()
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y


def load_ecoli_dataset():
    dataset = pd.read_csv('Datasets/ecoli.csv')
    dataset.drop(dataset[dataset['Localization Site'] == 'om'].index, inplace=True)
    dataset.drop(dataset[dataset['Localization Site'] == 'omL'].index, inplace=True)
    dataset.drop(dataset[dataset['Localization Site'] == 'imS'].index, inplace=True)
    dataset.drop(dataset[dataset['Localization Site'] == 'imL'].index, inplace=True)
    dataset.drop(columns=['Sequence Name'], inplace=True)
    dataset = dataset.to_numpy()
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    # sc = StandardScaler()
    # x = sc.fit_transform(x)
    le = LabelEncoder()
    y = le.fit_transform(y)
    return x, y


def load_dermatology_dataset():
    dataset = pd.read_csv('Datasets/dermatology.csv', header=None)
    dataset.iloc[:, 33].fillna(dataset.iloc[:, 33].median(), inplace=True)
    dataset = dataset.to_numpy()
    x = dataset[:, :-1]
    y = dataset[:, -1:].reshape(-1)
    sc = StandardScaler()
    x = sc.fit_transform(x)
    return x, y
