import math
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from scipy.spatial import distance


def getData():

    data = np.zeros((3200, 3))

    label = []

    with open('/Users/zahra_abasiyan/PycharmProjects/Project/deep_learning_course/datasets/SwissRoll.txt') as f:
        lines = f.readlines()

        for i in range(0, 3200):
            text = lines[i].split('\t')
            data[i][0] = text[0]
            data[i][1] = text[1]
            data[i][2] = text[2]
            label.append(int(text[3].replace('\n', '')))

    num_of_classes = len(set(label))

    X_train, X_test_tmp, y_train, y_test_tmp = train_test_split(data, label, test_size=0.30, random_state=42)

    X_validation, X_test, y_validation, y_test = train_test_split(X_test_tmp, y_test_tmp, test_size=0.33, random_state=42)

    return X_train, X_test, X_validation, y_train, y_test, y_validation, num_of_classes


def plot3D(data, label):

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(data[:,0], data[:,1], data[:,2], c=label)

    plt.show()

    return


def get_one_hot(Y):

    T = np.zeros((np.shape(Y)[0], len(set(Y))))

    for i in range(np.shape(Y)[0]):
        T[i, Y[i]] = 1

    return T


def getSigmaRadial():

    dist = []

    for center in centers:

        for center2 in centers:
            dist.append(np.sqrt(np.sum(np.subtract(center2, center) ** 2)))

    sigma = max(dist) / np.sqrt(2 * cluster_count)

    return sigma


def get_phi_rbf(X, center, sigma):

    return math.exp(np.linalg.norm(np.subtract(X, center) ,ord=2) / (-2*sigma))


def mahalanobis_dist(u, v, cov):

    dist = distance.mahalanobis(u, v, np.transpose(cov))

    return dist


def getHiddenInput_rbf(data):

    train_size = np.shape(data)[0]

    input = np.zeros((train_size, cluster_count))

    sigma = getSigmaRadial()

    for i in range(0, train_size):

        for j in range(0, cluster_count):

            input[i,j] = get_phi_rbf(data[i], centers[j], sigma)

    return input


def getHiddenInput_ebf(data):

    train_size = np.shape(data)[0]

    input = np.zeros((train_size, cluster_count))

    for i in range(0, train_size):

        for j in range(0, cluster_count):

            input[i,j] = get_phi_EBF(data[i], centers[j], cov_matrix[j])

    return input


def get_phi_EBF(x, center, cov):

    return math.exp((mahalanobis_dist(x, center,cov) ** 2) / (-2))


def setCovarianceOfEachCluster():

    cov_matrix = []

    for index in set(labels):

        count = 0
        tmp = np.zeros((dimension, dimension))
        i = 0
        tmp = np.matrix(tmp)
        for num in labels:

            if num == index:
                count += 1
                mat_tmp = np.matrix(x_train[i] - centers[index])
                mat =  np.transpose(mat_tmp) @ mat_tmp
                tmp += mat

            i += 1

        tmp /= count
        cov_matrix.append(tmp)

    return cov_matrix


x_train, x_test, x_validation, y_train, y_test, y_validation, num_of_classes = getData()

y_train_one_hot = get_one_hot(y_train)

y_test_one_hot = get_one_hot(y_test)

y_validation_one_hot = get_one_hot(y_validation)

dimension = np.shape(x_train)[1]

cluster_count = 1000

cls = KMeans(cluster_count)

cls.fit(x_train)

centers = cls.cluster_centers_

labels = (cls.labels_)

cov_matrix = setCovarianceOfEachCluster()

train_size = np.shape(x_train)[0]

plot3D(x_train, y_train)

X_train_rbf = (getHiddenInput_rbf(x_train))

X_test_rbf = (getHiddenInput_rbf(x_test))

X_validation_rbf = getHiddenInput_rbf(x_validation)

X_train_ebf = getHiddenInput_ebf(x_train)

X_test_ebf = getHiddenInput_ebf(x_test)

X_validation_ebf = getHiddenInput_ebf(x_validation)