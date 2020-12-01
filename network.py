#%%
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
# %matplotlib qt
import numpy as np
import pandas as pd
from scipy import sparse
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

    def Matrix_Multiply(self):
        """
        Here we try to obtain the clustering coefficient by multiplying the adjacency matrix A with itself,
        and then taking the multiplication A * A^2 only for those rows and columns that will form the diagonal of
        A^3, since those are the only values form A^3 that give the amount of closed paths of length 3 (i.e. triangles)
        The matrix multiplication method that is used here is just the standard numpy one. We need to find a better one.
        """
        diagonal = []
        print(self.A[1, 100])
        matrix2 = np.matmul(self.A, self.A)

        for i in range(self.A_size):
            value = 0
            for j in range(self.A_size):
                if (i != j):
                    continue
                for k in range(self.A_size):
                    print(matrix2[i, k])
                    value = value + self.A[i, k] * matrix2[k, i]
            diagonal.append(value)

        return diagonal



    # iv) degree distribution ===============================================================================================

    def plotDegDis(self, style):
        # DegreeDistribution dd
        # count how many nodes have 1, 2, 3 .. n degrees
        dd = [[degree, self.DegreeList.tolist().count(degree)] for degree in set(self.DegreeList)] # DegreeList is actually numpy.ndarray
        # split into two variables for plotting
        x = [dd[i][0] for i in range(len(dd))]
        y = [dd[i][1] for i in range(len(dd))]



        if style == 'linear':
            # plt.subplot(3, 1, 1)        # divides into 2 rows & 1 column, uses location 1 (ie top row)
            plt.bar(x[:49], y[:49])
            plt.title('Linear relationship')
            # plt.legend(loc = 'best')
            plt.xlabel('degrees (<=50)')
            plt.ylabel('number of nodes')
            plt.grid()                  # show grid
            plt.savefig('lin_plot.png', dpi = 200, transparent = True)
            plt.show()

        elif style == 'loglog':
            # plt.subplot(3,1,3)
            plt.plot(np.log(x), np.log(y), 'ko')
            plt.title('Log-log relationship')
            # plt.legend(loc = 'best')
            plt.xlabel('log(degrees)')
            plt.ylabel('log(number of nodes)')
            plt.savefig('loglog_plot.png', dpi = 200, transparent = True)
            plt.show()





    # v) average neighbour degree ===============================================================================================

    def AverageNeighbourDegree(self):
        randNode = np.random.randint(0, self.A_size)

        neighbours = np.where(self.A[randNode, :] == 1)[0]                    # returns tuple

        # what about a list comprehension?
        # dont think its possible without using different data format, eg pandas Series or DataFrame


        print("Node %d has %d neighbours (double-check: %d), which are located at %s." % (randNode, len(neighbours), self.DegreeList[randNode], neighbours))

        print("\nThose neighbours themself have on average %s neighbours." % round(np.add.reduce([self.DegreeList[i] for i in neighbours])/len(neighbours), 2))

        print("\nThe average number of degrees in the entire network is %s." % round(np.add.reduce(self.DegreeList)/len(self.DegreeList), 3))








#%% run once ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildFromData()



#%% run afterwards by importing ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildByImport()

# N1.visualise()




#%% iii) clustering coefficient ===============================================================================================
pass








#%% iv) probability mass function ===============================================================================================
N1.plotDegDis('linear')
N1.plotDegDis('loglog')






#%% v) avg degree of neighbours ===============================================================================================
N1.AverageNeighbourDegree()



