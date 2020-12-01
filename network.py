#%%
import time
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

#################### TO DO ##########################################

# scale free or non scale free? then choose another from https://snap.stanford.edu/data/index.html
# see log graph

# degree for all neighbours


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
        
        

    def visualise(self):
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







#%% run once ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildFromData()






#%% run afterwards by importing ===============================================================================================
n1 = NetworkBuilder("enron")
N1 = n1.buildByImport()

# N1.visualise()

N1.PlotDegreeDistribution()



#%% iii) clustering coefficient ===============================================================================================
pass








#%% iv) probability mass function ===============================================================================================
N1.PlotDegreeDistribution()







#%% v) avg degree of neighbours ===============================================================================================
N1.AverageNeighbourDegree()


