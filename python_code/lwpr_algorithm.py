"""
lwpr_algorithm.py: Uses the Locally Weighted Projection Regression (LWPR) based on Sethu Vijayakumar,
Aaron D'Souza and Stefan Schaal. Incremental Online Learning in High Dimensions. 
"""

from numpy import *
from lwpr import LWPR
from matplotlib import pyplot as plt
import socket


class LocalWeightedPR:

    def __init__(self):
        # amount of training epochs
        self.epochs = 100
        self.percentage_of_trest_data = 15.0

    def get_data(self):
        """
        Get training data from a file
        :return: List with unprocessed training data
        """
        # choose the dataset
        # full dataset 120k lines
        # path = r"full_dataset_120k.csv"
        # linear dataset
        # path = r"linear.csv"
        # linear dataset with plateau
        # path = r"linear_plateau.csv"
        # sin dataset
        # path = r"sinus_noise.csv"
        # own dataset started earlier to record and took only the first 100 lines and less 4,2k lines
        # path = r"harsha_evolution_cropped.csv"
        # new generated data from keras
        path = r"sharvar_keras_data.csv"
        return genfromtxt(path, delimiter=',')

    def train_model(self, model, Ntrain, Xtr, Ytr):
        """
        Train the model with given training data
        :param model: model of the lwpr algorithm non-trained
        :param Ntrain: number of training data lines
        :param Xtr: training data inputs
        :param Ytr: training data labels
        :return: a trained model with the given training data
        """

        # train the model
        plot_mse = []
        plot_rfs = []
        nMSE= 0
        for k in range(self.epochs):
            ind = random.permutation(Ntrain)
            mse = 0

            for i in range(Ntrain):
                yp = model.update(Xtr[ind[i]], Ytr[ind[i]])
                mse = mse + (Ytr[ind[i], :] - yp) ** 2

            nMSE = mse / Ntrain / var(Ytr)
            print "# %2i Data: %5i  #RFs: %3i  nMSE=%5.3f" % (k, model.n_data, model.num_rfs, nMSE)
            plot_mse.append(nMSE)
            plot_rfs.append(model.num_rfs)

        # Use to plot mse or rfs
        # plt.plot(plot_rfs)
        # plt.plot(plot_mse)
        # plt.show()

        return model

    def show_plot(self, Xtr, Ytr, test_set, model, dim):
        """
        Shows plots of the predictions of the trained model
        :param Xtr: training data inputs
        :param Ytr: training data labels
        :param test_set: data from the full dataset used for testing
        :param model: a trained model
        :param dim: amount of dimension of input
        """
        # test the model with unseen data
        Ntest = len(test_set)
        Xtest, Ytest = test_set[:, :dim], test_set[:, -1:]
        Xtest_sorted = []

        for element in Xtest:
            if Xtest.shape[1] > 1:
                Xtest_sorted.append(element)
            else:
                Xtest_sorted.append(float(element))

        if Xtest.shape[1] > 1:
            Xtest_sorted.sort(key=lambda x: x[0])
        else:
            Xtest_sorted.sort()

        Ytest = zeros((Ntest, 1))
        Conf = zeros((Ntest, 1))

        for k in range(Ntest):
            Ytest[k, :], Conf[k, :] = model.predict_conf(array([Xtest_sorted[k]]))

            # check for "classification"
            # if Ytest[k, :] >= 0:
            #     Ytest[k, :] = 1
            # else:
            #     Ytest[k, :] = -1
            # print Xtest[k], Ytest[k]

        plt.plot(Xtr, Ytr, 'r.')
        plt.plot(Xtest_sorted, Ytest, 'b-')
        plt.plot(Xtest_sorted, Ytest + Conf, 'c-', linewidth=2)
        plt.plot(Xtest_sorted, Ytest - Conf, 'c-', linewidth=2)
        plt.show()

    def networking(self, model):
        """
        Establishes a UDP connection with the script containing the simulation
        :param model: the trained model
        """
        UDP_IP = "127.0.0.1"
        UDP_PORT = 5005

        sock = socket.socket(socket.AF_INET,  # Internet
                             socket.SOCK_DGRAM)  # UDP
        sock.bind((UDP_IP, UDP_PORT))

        print "Socket set, waiting..."

        while True:
            data, addr = sock.recvfrom(1024)  # buffer size is 1024 bytes
            print "Received message:", data

            data = map(float, data[1:-1].split())  # gotta trim a bit, take care with [1:-1]

            if len(data) > 0:
                result = model.predict(array([data]))
                sock_send = socket.socket(socket.AF_INET,  # Internet
                                          socket.SOCK_DGRAM)  # UDP
                print "Send message:", str(result)
                sock_send.sendto(str(result), (UDP_IP, 5006))

    def all(self):
        """
        The main method bringing together all functionalities
        """
        data = self.get_data()

        # Cut into train and test set, based on percentage
        random.shuffle(data)
        train_set = data[int(len(data) * (self.percentage_of_trest_data / 100)):]
        test_set = data[:int(len(data) * (self.percentage_of_trest_data / 100))]  # first n

        print "Length full set: %2i" % data.shape[0]
        print "Length train set: %2i" % train_set.shape[0]
        print "Length test set: %2i" % test_set.shape[0]

        row, column = train_set.shape
        dim = column - 1
        Ntrain = row
        Xtr, Ytr = train_set[:, :dim], train_set[:, -1:]

        # initialize the LWPR model
        model = LWPR(dim, 1)
        model.init_D = 20 * eye(dim)
        model.update_D = True
        model.init_alpha = 40 * eye(dim)
        model.meta = False

        # train the model
        model = self.train_model(model, Ntrain, Xtr, Ytr)

        # show data
        # self.show_plot(Xtr, Ytr, test_set, model, dim)

        # establish udp connection etc
        self.networking(model)


if __name__ == '__main__':
    LocalWeightedPR().all()
