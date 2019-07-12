"""
main.py: Open gym implementation of cartpole, connection to the lwpr algorithm and containing an own
version of lienar regression and a sklearn linear regression

"""

import gym
import numpy as np
from sklearn import linear_model
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from scipy.spatial import distance_matrix
from scipy import stats
from Network import networking
import math
import socket
import gym.envs.classic_control

env = gym.make('CartPole-v1')
np.set_printoptions(suppress=True, precision=5)

UDP_IP = "127.0.0.1"
UDP_PORT_SEND = 5005


class CartPole:

    def __init__(self):
        # use_alg = 0: lwpr
        # use_alg = 1: own beta LR
        # use_alg = 2: sklearn
        self.use_alg = 0

        # File path for training data
        self.path = r"sharvar_keras_data.csv"
        # helper
        self.learned = 0
        self.learned_model_sk = 0

    def cartpole(self):
        """
        The main method starting the cartpole simulation
        :return:
        """
        while 1:
            observation = env.reset()

            # After 200 timesteps the environment is considered to be done
            for t in range(200):
                env.render()
                action = 0

                # use udp lwpr library
                if self.use_alg == 0:
                    networking.Network().send(observation)

                    try:
                        action = networking.Network().receive()
                    except socket.error:
                        print("No data yet")

                    action = float(action.decode("utf8")[1:-1])

                    # try to classification problem, by testing different limits
                    if action < 0.5:
                        action = 0
                    else:
                        action = 1

                    print(f"action {action}")

                # own linear regression
                elif self.use_alg == 1:
                    if self.learned:
                        prediction = beta_[0] \
                                     + beta_[1] * observation[0] \
                                     + beta_[2] * observation[1] \
                                     + beta_[2] * observation[2] \
                                     + beta_[2] * observation[3]

                        if prediction < 0.5:
                            action = 0
                        else:
                            action = 1
                    else:
                        data = np.genfromtxt(self.path, delimiter=',')
                        beta_ = self.learn_beta(data)
                        self.learned = 1

                # sk_learn
                elif self.use_alg == 2:
                    if self.learned:
                        self.learned_model_sk.predict(np.array([observation]))
                    else:
                        data = np.genfromtxt(self.path, delimiter=',')
                        self.learned_model_sk = self.learn_sk(data)
                        self.learned = 1

                observation, reward, done, info = env.step(action)
                if done:
                    break

    def learn_sk(self, observation):
        """
        sk_learn regression
        Problem: where to add weights?
        :param observation:
        :return:
        """
        observation = np.array(observation)
        reg = linear_model.LinearRegression()

        X, y = observation[:, :4], observation[:, -1:]

        # learn the model
        reg.fit(X, y)

        print("Regression coefficient: ", reg.coef_)
        return reg

    def learn_beta(self, observation):
        """
        Own implementation of linear regression
        :param observation:
        :return:
        """
        observation = np.array(observation)

        X, y = observation[:, :4], observation[:, -1:]
        X = self.prepend_one(X)

        print(np.array(X.T).shape)

        # for point in X:
        #     self.kernel(point, X, 0.5)

        # when weights are computed use np.diag() to put w_k into W
        W = np.identity(np.array(X.T).shape[1])  # place holder for weights (not computed yet)
        print("W.shape: ", W.shape)
        beta_ = mdot([inv(mdot([X.T, W, X])), X.T, W, y])
        print(f"Beta_: {beta_}")
        return beta_

    def kernel(self, point, xmat, k):
        get_shape = np.array(xmat).shape[0]

        # same as distance_matrix(xmat, xmat)
        # Use for verification, contivnue below
        # self.distance_matrix_own(get_shape, xmat)

        # get upper triangular distance matrix p.7 (5)
        M = np.triu(distance_matrix(xmat, xmat), 0)
        # print(f"M shape {M.shape} \n {M[:5,][:5,]}")

        # get positive definite distance matric p.7 (5)
        D = dot(M.T, M)
        # why is it different than the solution in the internet?
        # https: // matrixcalc.org / en /  # %7B%7B0,1,2,3%7D,%7B0,0,4,5%7D,%7B0,0,0,6%7D,%7B0,0,0,0%7D%7D%2A%7B%7B0,0,0,0%7D,%7B1,0,0,0%7D,%7B2,4,0,0%7D,%7B3,5,6,0%7D%7D
        # here: firts row and column are zero instead of last row and column
        # print(f"D.dot {D[:5,][:5,]}")
        w_k = np.zeros(np.array(xmat).shape[1])

        for j in range(get_shape):
            diff = point - xmat[j]
            print(f"diff {diff}")
            # w_k[j, j] = np.exp(diff * diff.T / (-2.0 * k ** 2))

            w_k[j, j] = math.exp(-(1/2) * mdot([diff.T, D, diff]))

    def distance_matrix_own(self, get_shape, xmat):
        """
        Return upper triangular distance matrix
        :param get_shape: shape of input matrix
        :param xmat: input matrix with x values
        :return:
        """
        M = np.zeros(shape=(get_shape, get_shape), dtype=float)
        print("D.shape: ", M.shape)
        for i in range(get_shape):
            for j in range(get_shape):
                dr = xmat[j] - xmat[i]  # difference between 2 positions
                M[i, j] = np.sqrt(sum(dr * dr))  # calculate distance and store
        M = np.triu(M, 0)
        print(f"D {D[:5, :5]}")
        return M

    def prepend_one(self, X):
        """prepend a one vector to X."""
        return np.column_stack([np.ones(X.shape[0]), X])


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.cartpole()
