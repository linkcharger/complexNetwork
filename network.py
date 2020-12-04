# %%
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.stats import poisson
from math import factorial
# from collections import Counter

#################### TO DO ##########################################

# scale free or non scale free? then choose another from https://snap.stanford.edu/data/index.html
# see log graph
# -> okay now what?

# bonus: estimation of fit to poisson (random graph) and power law (scale free graph)

######################################################################


# i) choose network ===============================================================================================

# enrom is ....?


# ii) adjacency matric A ===============================================================================================

# creational pattern: builder pattern
class NetworkBuilder:
    def __init__(self, data_file):
        self._data_file = data_file

    def buildFromData(self):

        # open file
        with open(self._data_file + ".txt", "r") as data:
            for i in range(4):
                data.readline()	  # skip meta info
            adjacency_list = data.readlines()
        print("\n-- file imported")

        length = len(adjacency_list)
        print("\n-- length adjacency list: " + str(length))

        # getting indices of non-zero items
        rowind = []
        colind = []
        for i in range(length):
            line = adjacency_list[i].split()
            rowind.append(int(line[0]))
            colind.append(int(line[1]))

        self._A_size = max(rowind)+1
        self._A = np.zeros((self._A_size, self._A_size), dtype='uint8')
        for i in range(length):
            rowindex = rowind[i]
            columnindex = colind[i]

            # setting non-zero items in full matrix
            self._A[rowindex][columnindex] = 1
        print("\n-- full matrix created")
        print("matrix size: " + str(self._A_size) + "^2")
        print(str(self._A[:6, :6]) + "\n")

        # create sparse matrix
        sparsevaluelist = [1] * (length * 1)  # vector of ones

        self._A_sparse = sparse.coo_matrix((sparsevaluelist, (rowind, colind)))
        print("\n-- sparse matrix created!")
        print(self._A_sparse.toarray()[:6, :6])

        # by summing the total amount of neighbours each vertex has
        self._DegreeList = np.add.reduce(self._A)
        print("\n-- degree list created: \n" + str(self._DegreeList))

        # export
        sparse.save_npz(str(self._data_file) + "-A_sparse.npz", self._A_sparse)
        print("\n-- exported as npz")

        #
        #

        return Network(self._A, self._A_sparse, self._A_size, self._DegreeList)

    def buildByImport(self):

        self._A_sparse = sparse.load_npz(self._data_file + "-A_sparse.npz")
        print("\n-- A_sparse imported from npz")
        self._A = self._A_sparse.toarray()
        print("\n-- A created from A_sparse")
        self._A_size = len(self._A)
        print("\n-- A_size calculated")
        self._DegreeList = np.add.reduce(self._A)
        print("\n-- degree list created: \n" + str(self._DegreeList))

        #
        #

        return Network(self._A, self._A_sparse, self._A_size, self._DegreeList)


########################################################################################################################
########################################################################################################################

class Network:

    def __init__(self, A, A_sparse, A_size, DegreeList):
        self.A = A
        self.A_sparse = A_sparse
        self.A_size = A_size
        self.DegreeList = DegreeList

    # def visualise(self):
        # thought that would be nice to have for the project documentation in the end..
        # maybe now it will work with the sparse matrix

        # fig = plt.figure()
        # ax = fig.add_subplot(121)
        # ax.imshow(self.A, interpolation='bilinear', cmap=cm.Greys_r)
        # plt.show()

        # plt.spy(self.A, precision=0)

    # iii) average clustering coefficient ===============================================================================================

    def GetMeanC(self):
        """
       Function to obtain clustering coefficient
        """

        diagonal = []
        A2_sparse = self.A_sparse @ self.A_sparse
        A3_sparse = A2_sparse @ self.A_sparse
        print("Matrix multiplication done.")
        A3_dense = A3_sparse.toarray()

        for i in range(self.A_size):
            diagonal.append(A3_dense[i, i])

        print("Filled the diagonal list.")
        C = 0

        for i in range(self.A_size):
            #print("diagonal has value ", diagonal[i], "while degreelist has value ", self.DegreeList[i])
            if diagonal[i] == 0:
                continue
            C = C + diagonal[i]/(self.DegreeList[i]*(self.DegreeList[i] - 1))

        C_mean = C/self.A_size

        print("Mean clustering was found to be %lf" % (C_mean))

    # iv) degree distribution ===============================================================================================

    def plotDegDis(self, style):
        # DegreeDistribution dd
        # count how many nodes have 1, 2, 3 .. n degrees
        dd = [[degree, self.DegreeList.tolist().count(degree)] for degree in set(
            self.DegreeList)]  # DegreeList is actually numpy.ndarray
        # split into two variables for plotting
        x = [dd[i][0] for i in range(len(dd))]
        y = [dd[i][1] for i in range(len(dd))]
        weight = np.add.reduce(y)
        y_weighted = [float(i/weight) for i in y]

        if style == 'linear':
            # plt.subplot(3, 1, 1)        # divides into 2 rows & 1 column, uses location 1 (ie top row)
            plt.bar(x[:49], y_weighted[:49])
            plt.title('Linear relationship')
            # plt.legend(loc = 'best')
            plt.xlabel('degrees (<=50)')
            plt.ylabel('number of nodes')
            plt.grid()                  # show grid
            plt.savefig('lin_plot.png', dpi=200, transparent=True)
            plt.show()

        elif style == 'loglog':
            # plt.subplot(3,1,3)
            plt.plot(np.log(x), np.log(y_weighted), 'ko')
            plt.title('Log-log relationship')
            # plt.legend(loc = 'best')
            plt.xlabel('log(degrees)')
            plt.ylabel('log(number of nodes)')
            plt.savefig('loglog_plot.png', dpi=200, transparent=True)
            plt.show()

    # v) average neighbour degree ===============================================================================================

    def AverageNeighbourDegree(self):
        randNode = np.random.randint(0, self.A_size)

        neighbours = np.where(self.A[randNode, :] == 1)[
            0]                    # returns tuple

        Mean_Degree = 0

        for i in range(self.A_size):
            neighbours = np.where(self.A[i, :] == 1)[0]

            Mean_Degree = Mean_Degree + \
                round(np.add.reduce([self.DegreeList[j]
                                     for j in neighbours])/len(neighbours), 2)
        Mean_Degree = Mean_Degree/self.A_size

        # what about a list comprehension?
        # dont think its possible without using different data format, eg pandas Series or DataFrame

        #print("Node %d has %d neighbours (double-check: %d), which are located at %s." % (randNode, len(neighbours), self.DegreeList[randNode], neighbours))

        #print("\nThose neighbours themself have on average %s neighbours." % round(np.add.reduce([self.DegreeList[i] for i in neighbours])/len(neighbours), 2))

        print("\nThe average amount of neighbour's neighbours is %s." %
              round((Mean_Degree), 3))

        print("\nThe average number of degrees in the entire network is %s." %
              round(np.add.reduce(self.DegreeList)/len(self.DegreeList), 3))

    # vii & viii) Fitting to Poisson and power law ===============================================================================

    def Fitting(self):

        dd = [[degree, self.DegreeList.tolist().count(degree)] for degree in set(
            self.DegreeList)]  # DegreeList is actually numpy.ndarray
        # split into two variables for plotting
        x = [dd[i][0] for i in range(len(dd))]
        y = [dd[i][1] for i in range(len(dd))]
        weight = np.add.reduce(y)
        y_weighted = [float(i/weight) for i in y]

        def Poisson(k, lamb):
            return poisson.pmf(k, lamb)


        def Powerlaw(x, a, b):
            return a*x**b


        params_poisson, c_poisson = curve_fit(Poisson, x, y_weighted)

        lamb = params_poisson

        params_power, c_power = curve_fit(Powerlaw, x, y_weighted)

        a, b = params_power
        
        plt.figure(0)
        axes = plt.axes()
        axes.set_xlim([0,50])
        plt.plot(x, Poisson(x, lamb))

        print("We have lambda = %.5lf" %lamb)

        plt.figure(1)
        axes = plt.axes()
        axes.set_xlim([0,50])
        plt.plot(x, Powerlaw(x, a, b))

        print("We have a = %.5lf and b = %.5lf" % (a, b))
        
        


# %% run once ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildFromData()

# %% run afterwards by importing ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildByImport()
# N1.visualise()

# %% iii) clustering coefficient ===============================================================================================
N1.GetMeanC()

# %% iv) probability mass function ===============================================================================================
N1.plotDegDis('linear')
N1.plotDegDis('loglog')

# %% v) avg degree of neighbours ===============================================================================================
N1.AverageNeighbourDegree()

# %% vii & viii) Fitting to Poisson and power law
N1.Fitting()
