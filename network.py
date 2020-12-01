
# %%
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

#################### TO DO ##########################################

# scale free or non scale free? then choose another from https://snap.stanford.edu/data/index.html

# clustering coefficient:
	# fast matrix multiplication algorithm
	# which equation/approach?


# Bonus is all theoretical;

######################################################################


# i) choose network ===============================================================================================

# enrom is ....?


# ii) adjacency matric A ===============================================================================================

class Network:

    def __init__(self, data_file):
        self.data_file = data_file

        # open file
        with open(data_file + ".txt", "r") as self.data:
            for i in range(4):
                self.data.readline()	  # skip meta info
            adjacency_list = self.data.readlines()

        print("-- file imported")

        length = len(adjacency_list)
        print("length adjacency list: " + str(length))

        # getting indices of non-zero items
        rowind = []
        colind = []
        for i in range(length):
            line = adjacency_list[i].split()
            rowind.append(int(line[0]))
            colind.append(int(line[1]))

        self.A_size = max(rowind)+1
        self.A = np.zeros((self.A_size, self.A_size), dtype=int)

        for i in range(length):
            rowindex = rowind[i]
            columnindex = colind[i]

            # setting non-zero items in full matrix
            self.A[rowindex][columnindex] = 1

        print("-- full matrix created")
        print("matrix size: " + str(self.A_size) + "^2")
        print(str(self.A[:6, :6]) + "\n")

        # create and fill sparse matrix
        sparsevaluelist = [1] * (length * 1)  # vector of ones

        self.A_sparse = sparse.coo_matrix((sparsevaluelist, (rowind, colind)))
        print("-- sparse matrix created!")
        # print(self.A_sparse.toarray()) #[:6, :6]

        

    def exportA_sparse(self):
        sparse.save_npz(str(self.data_file) + "A_sparse.npz", self.A_sparse)
        print("-- exported as npz")

    def importA_sparse(self):
        self.A_sparse = sparse.load_npz(self.data_file + "A_sparse.npz")
				self.A = self.A_sparse.toarray()


    
    def visualisation(self):
        # thought that would be nice to have for the project documentation in the end..
            # maybe now it will work with the sparse matrix

        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow(self.A, interpolation='bilinear', cmap=cm.Greys_r)
        plt.show()

        plt.spy(self.A, precision=0)

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

    def PlotDegreeDistribution(self):
        # Sums the total amount of neighbours each vertex has and saves that value in DegreeList
        self.DegreeList = np.add.reduce(self.A)
        print("-- degree list: " + str(self.DegreeList))

        # degree distribution as a histogram
        bins = np.linspace(0, 70, 70)
        plt.hist(x=self.DegreeList, bins=bins)
        plt.show()


# v) average neighbour degree ===============================================================================================

    def AverageNeighbourDegree(self):
        index = np.random.randint(0, self.A_size)
        collist = []
        col = 0

        for i in self.A[index, :]:
            if i != 0:
                collist.append(col)
            col = col + 1

        print("Vertex %d has %d neighbours, which are located at at" %
              (index, self.DegreeList[index]))
        print(collist)

        AverageDegree = 0
        for i in collist:
            AverageDegree = AverageDegree + self.DegreeList[i]
        AverageDegree = AverageDegree/len(collist)
        print("The average degree found is %lf" % (AverageDegree))


# %% create once ===============================================================================================
N1 = Network("enron")
N1.exportA_sparse()

# %% import and use ===============================================================================================
# N1 = Network("enron")
N1.importA_sparse()                         # problem: re-creates the network object and its properties from scratch, not from the imported data


N1.PlotDegreeDistribution()
N1.AverageNeighbourDegree()