# %%
import time
from math import exp, factorial

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gammaln
from scipy.stats import chisquare
from math import factorial
from math import exp
#from collections import Counter




# To see whether everything would work on a different network, I tried this one: https://snap.stanford.edu/data/ca-AstroPh.html
# I think there's a problem with the renumbering. When trying to find the mean C for the astro graph, there are about 10 nodes which say they have 1 closed triangle, with only 1 neighbour. This is impossible.
# Ignoring these +- 10 nodes, we do find the average clustering coefficient we're supposed to find, so we can probably assume that these 10 are the only ones that are annoying us.
# The fitting to the Poisson distribution is very probably incorrect. I have not found the correct way of doing this yet.
# Looking at the log log distribution of the Astro graph, it does seem that this is a network with scale.

#################### TO DO ##########################################

# scale free or non scale free? then choose another from https://snap.stanford.edu/data/index.html
# Improve fitting to the Poisson distribution
# Find out what is going wrong in the renumbering of the astro network
# Do a good reduced chisquared test. I think my version of the test is not working correctly

######################################################################


# i) choose network ===============================================================================================

# enrom is ....?


# ii) adjacency matric A ===============================================================================================

########################################################################################################################
########################################################################################################################

# creational pattern: builder pattern
class NetworkBuilder:
    def __init__(self, data_file, mode = ""):
        self._data_file = data_file
        self.mode = mode

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

        if self.mode == "renumber":
            new_rowind = renumber(rowind)
            rowind = new_rowind
            new_colind = renumber(colind)
            colind = new_colind

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
        sparse.save_npz(str(self._data_file) + " - A_sparse.npz", self._A_sparse)
        print("\n-- exported as npz")

        #
        #

        return Network(self._data_file, self._A, self._A_sparse, self._A_size, self._DegreeList)

    def buildByImport(self):

        self._A_sparse = sparse.load_npz(self._data_file + " - A_sparse.npz")
        print("\n-- A_sparse imported from npz")
        self._A = self._A_sparse.toarray()
        print("\n-- A created from A_sparse")
        self._A_size = len(self._A)
        print("\n-- A_size calculated")
        self._DegreeList = np.add.reduce(self._A)
        print("\n-- degree list created: \n" + str(self._DegreeList))

        for i in range(self._A_size):
            if(self._A[i,i] == 1):
                self._A[i,i] = 0
                self._DegreeList[i] = self._DegreeList[i] - 1
        
        self._A_sparse = sparse.coo_matrix(self._A)

        #
        #

        return Network(self._data_file, self._A, self._A_sparse, self._A_size, self._DegreeList)


########################################################################################################################
########################################################################################################################

class Network:

    def __init__(self, name, A, A_sparse, A_size, DegreeList):
        self.name = name
        self.A = A
        self.A_sparse = A_sparse
        self.A_size = A_size
        self.DegreeList = DegreeList



    def showA(self):
        plt.figure(figsize = (8,8))
        plt.title(self.name + " - adjacency matrix")
        plt.spy(self.A_sparse, markersize = .005)
        plt.savefig(self.name + ' - plot_' +  'A.png', dpi=200, bbox_inches = 'tight')
        plt.close()



# iii) average clustering coefficient ===============================================================================================

    def GetMeanC(self):
        """
       Function to obtain clustering coefficient
        """

        diagonal = []
        C = 0

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
            if self.DegreeList[i] == 1 & diagonal[i] != 0:
                continue

            if diagonal[i] == 0:         
                continue
            C = C + diagonal[i]/(self.DegreeList[i]*(self.DegreeList[i] - 1))

        C_mean = C/(self.A_size)

        print("Mean clustering was found to be %lf" % (C_mean))

# iv) degree distribution ===============================================================================================

    def plotDegDis(self, style):
        # DegreeDistribution dd
        # count how many nodes have 1, 2, 3 .. n degrees
        dd = [[degree, self.DegreeList.tolist().count(degree)] for degree in set(self.DegreeList)]  # DegreeList is actually numpy.ndarray
        # split into two variables for plotting
        x = [dd[i][0] for i in range(len(dd))]
        y = [dd[i][1] for i in range(len(dd))]
        weight = np.add.reduce(y)
        y_weighted = [float(i/weight) for i in y]

        if style == 'linear':
            plt.figure(figsize = (10,6))        
            plt.bar(x[:49], y_weighted[:49])
            plt.title(self.name + ' - Degree distribution, linear scale')
            plt.xlabel('degrees (<=50)')
            plt.ylabel('number of nodes')
            plt.savefig(self.name + ' - plot_DD_lin.pdf', dpi=200, bbox_inches = 'tight')
            # plt.show()
            plt.close()

        elif style == 'loglog':
            plt.figure(figsize = (10,6)) 
            plt.plot(np.log(x), np.log(y_weighted), 'ko')
            plt.title(self.name + ' - Degree distribution, loglog scale')
            plt.grid()                                          # show grid
            plt.xlabel('log(degrees)')
            plt.ylabel('log(number of nodes)')
            plt.savefig(self.name + ' - plot_DD_loglog.pdf', dpi=200, bbox_inches = 'tight')
            # plt.show()
            plt.close()

# v) average neighbour degree ===============================================================================================

    def AverageNeighbourDegree(self):
        # randNode = np.random.randint(0, self.A_size)
        # neighbours = np.where(self.A[randNode, :] == 1)[0]                    # returns tuple
        # print("Node %d has %d neighbours (double-check: %d), which are located at %s." % (randNode, len(neighbours), self.DegreeList[randNode], neighbours))
        # print("\nThose neighbours themself have on average %s neighbours." % round(np.add.reduce([self.DegreeList[i] for i in neighbours])/len(neighbours), 2))


        Mean_Degree = 0
        for i in range(self.A_size):
            neighbours = np.where(self.A[i, :] == 1)[0]
            N = 0
            if (len(neighbours) == 0):
                continue
        
            Mean_Degree = Mean_Degree + np.add.reduce([self.DegreeList[j] for j in neighbours])/len(neighbours)
        Mean_Degree = Mean_Degree/self.A_size

        print("\nThe average amount of neighbour's neighbours is %s." %
              round((Mean_Degree), 3))

        print("\nThe average number of degrees in the entire network is %s." %
              round(np.add.reduce(self.DegreeList)/len(self.DegreeList), 3))


		
# vii & viii) Fitting to Poisson and power law ===============================================================================

    def Fitting(self):

        # Maybe we should combine this with the plotdegdis function, since we actually need everything that's in there for this function

        dd = [[degree, self.DegreeList.tolist().count(degree)] for degree in set(self.DegreeList)]  # DegreeList is actually numpy.ndarray
        # split into two variables for plotting
        x = np.array([dd[i][0] for i in range(len(dd))])
        y = np.array([dd[i][1] for i in range(len(dd))])
        weight = np.add.reduce(y)
        y_weighted = [float(i/weight) for i in y]

        def taj(x):
            return 1./x, 1./x**2.

        def Poisson(k, lamb):
            y, J = taj(k)
            return np.exp(y * np.log(lamb) - lamb - gammaln(y + 1.)) * J


        """ def Poisson(k, lamb):
            return poisson.pmf(k, lamb) """


        def Powerlaw(x, a, b):
            return a*x**b


        params_poisson, _ = curve_fit(Poisson, x, y_weighted)

        lamb = params_poisson

        params_power, _ = curve_fit(Powerlaw, x, y_weighted)

        a, b = params_power
        
        plt.figure(0)
        axes = plt.axes()
        axes.set_xlim([0,50])
        plt.bar(x[:49], y_weighted[:49])
        plt.plot(x, Poisson(x, lamb), 'r')

        print("We have lambda = %.5lf" %lamb)

        plt.figure(1)
        axes = plt.axes()
        axes.set_xlim([0,50])
        plt.bar(x[:49], y_weighted[:49])
        plt.plot(x, Powerlaw(x, a, b), 'r')

        print("We have a = %.5lf and b = %.5lf" % (a, b))


        poissonlist = [Poisson(i, lamb)[0] for i in x]
        powerlist = [Powerlaw(i, a, b) for i in x]

        chisq1 = chisquare(y_weighted, poissonlist)
        chisq2 = chisquare(y_weighted, powerlist)
        
        print(chisq1)
        print(chisq2)
        
        
########################################################################################################################
########################################################################################################################

def renumber(old_list):
    """
    This function removes all duplicates from a list and then sorts from lowest to highest number.
    Subsequently, the numbers in old_list are replaced with numbers from 0 to n, depending on where they are in the 
    ranking from lowest to highest number.
    """
    new_list = old_list
    no_duplicates = sorted(set(old_list))
    length = len(no_duplicates)
    no_duplicates = np.array(no_duplicates)

    for a in range(length):
        print("Now replacing the number of node %d of %d.\n" %(a, length))
        for n, i in enumerate(new_list):
            if i == no_duplicates[a]:
                new_list[n] = a
                
    return new_list


########################################################################################################################
########################################################################################################################

# %% just graphics
n1 = NetworkBuilder("Enron emails")
N1 = n1.buildByImport()
N1.showA()
N1.plotDegDis('linear')
N1.plotDegDis('loglog')

n2 = NetworkBuilder("Astrophysics citations")
N2 = n2.buildByImport()
N2.showA()
N2.plotDegDis('linear')
N2.plotDegDis('loglog')



# %% ENRON  ===============================================================================================
n1 = NetworkBuilder("Enron emails")
N1 = n1.buildFromData()
N1.showA()

# %% run afterwards by importing 
n1 = NetworkBuilder("Enron emails")
N1 = n1.buildByImport()
N1.showA()




# %% iii) clustering coefficient 
N1.GetMeanC()

# %% iv) probability mass function 
N1.plotDegDis('linear')
N1.plotDegDis('loglog')

# %% v) avg degree of neighbours 
N1.AverageNeighbourDegree()

# %% vii & viii) Fitting to Poisson and power law 
N1.Fitting()






# %% astrophysics citations =============================================================================
n2 = NetworkBuilder("Astrophysics citations", "renumber")
N2 = n2.buildFromData()
N2.showA()

# %% run afterwards by importing 
n2 = NetworkBuilder("Astrophysics citations")
N2 = n2.buildByImport()
N2.showA()

# %% iii) clustering coefficient 
N2.GetMeanC()

# %% iv) probability mass function 
N2.plotDegDis('linear')
N2.plotDegDis('loglog')

# %% v) avg degree of neighbours 
N2.AverageNeighbourDegree()
# %% vii & viii) Fitting to Poisson and power law 
N2.Fitting()
