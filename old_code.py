""" The basic element, representing the entity at each
    individual grid cell. Will be overridden to make other
    games of Life. """
class Cell:
    def __init__(self, val=None, pr=0.5):
        """ Initializes a cell with the specified value, or a random one if no
        value is provided."""
        if val == None:
            self.val = Cell.genRandVal(pr)
        else:
            self.val = val
        
    
        
    def __str__(self):
        return str(self.val)
        
    def getColor(self):
        """ Returns the "color" of this object - black if 1, white if 0. """
        if self.val == 1:
            return "black"
        else:
            return "white"
            
    @staticmethod
    def genRandVal(pr):
        """ Returns a random value for a Cell object (with probability pr of
            being alive); the default is either a 0 or 1 """
        if rand.random() < pr:
            return 1
        else:
            return 0
            
    @staticmethod
    def gridToStr(grid, dim):
        if len(dim) == 0:
            # only one cell left, so grid is a Cell, not a grid
            return str(grid)
        
        s = "["
        for i in range(dim[0]):
            s += Game.gridToStr(grid[i], Game.rmFirst(dim))
            if i != dim[0] - 1:
                # don't add a comma after the last array element
                s += ", "
        s += "]"
        return s
     
    @staticmethod
    def rmFirst(t):
        """ Removes the first element of a tuple. """
        return tuple(t[i] for i in range(1, len(t)))
        
        
    
    @autojit
    def evolve2D(self):
        """ The original evolve function of the game of life. Assumes possible
            states are 0 (dead) and 1 (alive), and that the grid is 2D. """
        if len(self.dim) != 2:
            raise ValueError("ERROR: evolve2D only works with 2D grids.")
        # copy the grid so that further changes aren't decided by previous ones
        gr = deepcopy(self.grid)
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                numAlive = 0
                for adj in self.adjGrid[i,j]:
                    numAlive += gr[adj]
                if numAlive < 2 or numAlive > 3:
                    self.grid[i,j] = 0
                elif numAlive == 3 and self.grid[i,j] == 0:
                    self.grid[i,j] = 1
                    
                    
""" Like evolve, but only compatible with 2D arrays. Uses loops rather than
        iterators, so hopefully easier to parallelize. Assumes grid and adjGrid
        are what they should be for dim = dimArr[0:1] (AND ARE CONFIGURED.)
        dimArr is [rows, cols, maxLen]"""
rows = grid.shape[0] - 1
maxLen = adjGrid.shape[2]
cols = grid.shape[1] - 1
# we'll only parallelize two dimensions - the third for loop may be quite
# short, and thus not worth parallelizing as some cores may be idle.
startX, startY = cuda.grid
gridX = cuda.gridDim.x * cuda.blockDim.x;
gridY = cuda.gridDim.y * cuda.blockDim.y;
for i in range(startX, rows, gridX):
    for j in range(startY, cols, gridY):
        numAlive = 0
        for k in range(maxLen):
            # if adjGrid is configured, a placeholder value of (-1, -1) will
            # result in a 0 being looked up in grid.
            numAlive += grid[adjGrid[i,j,k,0], adjGrid[i,j,k,1]]

        if numAlive == 3 or (numAlive == 2 and grid[i,j] == 1):
            newGrid[i+1,j+1] = 1
                
                
                
                
 

def countLiveCells(grid, adjGrid, i, j, maxLen):
    numAlive = 0
    for k in range(maxLen):
        # if adjGrid is configured, a placeholder value of (-1, -1) will
        # result in a 0 being looked up in grid.
        numAlive += grid[adjGrid[i,j,k,0], adjGrid[i,j,k,1]]
    return numAlive
    
countLiveCells_GPU = cuda.jit(restype=uint8, argtypes=[uint8[:,:], 
uint16[:,:,:,:], uint16, uint16, uint16], device=True)(countLiveCells)


def configure(grid, adjGrid):
    """ Configures grid and adjGrid for higher efficiency, i.e no using that
        troublesome if statement. """
    dim = addToTuple(grid.shape, 1)
    newGrid = np.zeros(dim, dtype=np.int8)
    it = np.nditer(grid, flags=['multi_index'], op_flags=['readonly'])
    while not it.finished:
        newGrid[addToTuple(it.multi_index, 1)] = grid[it.multi_index]
        it.iternext()
    
    newAdjGrid = np.empty_like(adjGrid)
    it = np.nditer(grid, flags=['multi_index'], op_flags=['readonly'])
    while not it.finished:
        newAdjGrid[it.multi_index] = adjGrid[it.multi_index] + 1
        it.iternext()
    return (newGrid, newAdjGrid)

def convAdjGrid(adjGrid, dim):
    """ Converts the adjacency grid from a Numpy object array to the much more
        efficient int32 array (which supports grids up to 32767 rows or columns)
        
        So as to enable Numpy to store this array as an array of integers,
        rather than objects (in particular, lists of tuples), "placeholder"
        values of [-1, -1] are inserted - this allows Numpy to use int8
        data-type, but the placeholder values have to be discounted.
        
        This method needs to be run after all adjGrid conversions have been
        completed.
    """
    # size of adjGrid, not including internal arrays
    size = adjGrid.shape
    # maximum length - this will be incorporated into the new shape
    maxLen = 0
    it = np.nditer(adjGrid, flags=['multi_index', 'refs_ok'],
        op_flags=['readonly'])
    while not it.finished:
        if maxLen < len(adjGrid[it.multi_index]):
            maxLen = len(adjGrid[it.multi_index])
        it.iternext()
    # number of elements in each tuple is the number of dimensions
    newGrid = np.full(size + (maxLen, len(dim)), -1, dtype=np.int32)
    it = np.nditer(adjGrid, flags=['multi_index', 'refs_ok'],
        op_flags=['readonly'])
    while not it.finished:
        for adjPos in range(len(adjGrid[it.multi_index])):
            for coord in range(len(dim)):
                # copy element over to new grid
                newGrid[it.multi_index][adjPos][coord] = \
                    adjGrid[it.multi_index][adjPos][coord]
        it.iternext()
    return newGrid

def rmFirst(t):
    """ Removes the first element of a tuple. """
    return tuple(t[i] for i in range(1, len(t)))


def getEdgeArr(adjGrid):
    """ Generates an array of all edges.
        Represented as arrays of two locations,which are also represented with
        arrays.
    """
    dim = adjGrid.shape
    # this will be the number of edges
    prod = 1
    for i in range(len(dim) - 1):
        prod *= dim[i]

    edges = np.zeros((prod, 2, dim[len(dim) - 1]))
    ind = 0 # current index in edges array

    # iterate over this array; we don't want to iterate into coordinates
    iter_arr = np.zeros(dim[0:len(dim)-1])
    it = np.nditer(iter_arr, flags=['multi_index'])
    while not it.finished:
        edges[ind][0] = it.multi_index

        it.iternext()