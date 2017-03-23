import math

def _euclidean_distance(p1, p2, dimension):
    val = 0
    for i in range(0,dimension):
        val += math.pow(p1[i] - p2[i], 2)
    return math.sqrt(val)

def knn(training_data, test_data, k, returns_data=False):
    dim = len(training_data[0][0])
    nbc = len(training_data)
    nn = []

    for cl in range(0,nbc): #for each different classes
        for data in training_data[cl]: #for each data in a class
            nn.append([cl,_euclidean_distance(test_data, data, dim),data])

    nn.sort(key=lambda x: x[1])
    sum_class = [0]*nbc

    for i in range(0,k): # calculate which class is more represented
        sum_class[nn[i][0]] += 1

    if returns_data:
        return [sum_class.index(max(sum_class)),nn]
    else:
        return sum_class.index(max(sum_class))
