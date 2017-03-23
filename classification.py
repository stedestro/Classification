import math

def _euclidean_distance(p1, p2):
    '''
    Calculate the Euclidean disstance between two points

    Arguments:
        p1,p2: Point in list [x,y,z] or tuple format (x,y,z)

    Return
        (float) Distance
    '''
    dim = len(p1)
    if dim != len(p2):
        raise Exception("Points dimension must match")
    val = 0
    for i in range(0,dim):
        val += math.pow(p1[i] - p2[i], 2)
    return math.sqrt(val)

def knn(training_data, test_data, k, returns_data=False):
    '''
    Calculate the k closest neighbours of a point. Classification is done by
    checking which class of data is more reprensented

    Arguments:
        training_data (List[List[]]) : The first level contains the different
            classes to test against, the second level contains an arbitrary
            number of points. The points are in tuple or list format
            (e.g. (x,y,z) or [x,y,z])
        test_data (point) : Point in list or tuple format (see above)
        k (int) : number of neighbours used to classify test_data
        returns_data (bool) : if True, add distances to the function's returns
            otherwise returns the index of calculated class

    Return:
        (int) index of calculated class
        (list) if returns_data is set to True,
            returns [(int)index of calculated[[(int)class, (float)distance, (point)],..]]
    '''
    nbc = len(training_data) # number of classes
    nn = [] # list containing all distances

    for cl in range(0,nbc): #for each different classes
        for data in training_data[cl]:
            nn.append([cl,_euclidean_distance(test_data, data),data])

    nn.sort(key=lambda x: x[1])
    sum_class = [0]*nbc
    for i in range(0,k): # calculate which class is more represented
        sum_class[nn[i][0]] += 1
    if returns_data:
        return [sum_class.index(max(sum_class)),nn]
    else:
        return sum_class.index(max(sum_class))
