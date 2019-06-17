"""

Basiert auf: show_basic_own.py, anfang des skripts wurde genutzt.

Aktueller Stand:
    - Lineare Regression aus ML VL ist implementiert
    - Receptive Fields angefangen

Aktuelle TODO generell:
    - Datensatz generieren
    
Aktuelle TODO (Matlab-Mix Ansatz):
    - Das Matlab Skript nutzen, um den CartPole zu steuern

"""

import gym
import numpy as np
from sklearn import linear_model
from numpy import dot
from numpy.linalg import inv
from numpy.linalg import multi_dot as mdot
from scipy.spatial import distance_matrix
from scipy import stats
import math

env = gym.make('CartPole-v1')
np.set_printoptions(suppress=True, precision=2)

class CartPole:

    def cartpole(self):
        highest_timestep = 0
        save_episode = 0
        learn_counter = 0
        observation_temp = []
        learned = False
        learned_model = linear_model.LinearRegression()

        for i_episode in range(20):
            observation = env.reset()
            for t in range(100):
                env.render()

                if not learned:
                    action = env.action_space.sample()
                else:
                    # used with learn_sk()
                    # action = learned_model.predict(np.array([observation]))

                    # used for 'saved' models
                    my_array = np.loadtxt('save_beta.csv', delimiter=",", skiprows=0)
                    saved_prediction = my_array[0] \
                                       + my_array[0] * observation[0] \
                                       + my_array[0] * observation[1] \
                                       + my_array[0] * observation[2] \
                                       + my_array[0] * observation[3]

                    # used with learn_beta()
                    prediction = beta_[0] \
                             + beta_[1] * observation[0] \
                             + beta_[2] * observation[1] \
                             + beta_[2] * observation[2] \
                             + beta_[2] * observation[3]


                    # show difference between sklearn and own algorithm
                    # print(f"Sk_learn vs Own: {action[0][0]:.1f} vs {prediction[0]:.1f}")

                    # print(f"Own prediction {prediction[0]:.01f}")

                    # define threshold for action?
                    if prediction < 0.5:
                        action = 0
                    else:
                        action = 1

                observation, reward, done, info = env.step(action)

                observation_row = []
                for element in observation:
                    observation_row.append(element)
                observation_row.append(action)
                observation_temp.append(observation_row)

                if done:
                    if t > 20 and t > highest_timestep:
                        if t >= 75:
                            try:
                                np.savetxt('save_beta.csv', beta_, delimiter=',')
                            except UnboundLocalError:
                                print("beta not bound yet")
                        highest_timestep = t
                        save_episode = i_episode
                        learn_counter += 1
                        learned = True

                        beta_ = self.learn_beta(observation_temp)

                        # learned_model = self.learn_sk(observation_temp)
                        # learned_model.predict(np.array([observation]))

                    observation_temp = []
                    print("Episode {} finished after {} timesteps".format(i_episode, t+1))
                    break
        print(f"Highest timestep {highest_timestep} in episode {save_episode}. Learned {learn_counter} times")
        env.close()

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
        REturn upper triangular distance matrix
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
