import math
import random

class Node:

    def __init__(self, dimension, data):
        self.left = None
        self.right = None
        self.dimension = dimension 
        self.data = data
        #print("{} - {}".format(self.data,self.dimension))
    
    def insert_node(self, data, dimension):
        if data:
            if data < self.data:
                self.left = Node(dimension, data)
                return self.left
            if data >= self.data:
                self.right = Node(dimension, data)
                return self.right  

    def insert_leaf(self, data, isMinor):
        if isMinor:
            self.left = Node(-1,data)
        else:
            self.right = Node(-1,data)

    def print_tree(self):
        print(self.data)

def create_kd_tree(data, k):
    dimension_len = len(data[0])
    return create_kd_tree_aux(data, dimension_len, k, None)
    

def create_kd_tree_aux(data, dimension_len, k, root):
    if k<2:
        k=2
    median=0
    random_dimension = random.randint(0, dimension_len-2)
    if root is None:
        for x in data:
            median += x[random_dimension]
        median = median / len(data)
        root = Node(random_dimension, median)
        new_data = split_data(random_dimension,data,median)
        #print("L")
        create_kd_tree_aux(new_data[0], dimension_len, k, root)
        #print("R")
        create_kd_tree_aux(new_data[1], dimension_len, k, root)
        return root;
    elif len(data)<=k:
        if len(data)!=0:
            if root.data > data[0][root.dimension]:
                #print("L")
                root.insert_leaf(data,True)
            else:
                #print("R")
                root.insert_leaf(data,False)
        else:
            root.insert_leaf(None,True)
            root.insert_leaf(None,False)
    else:
        for x in data:
            median += x[random_dimension]
        median = median / len(data)
        root=root.insert_node(median,random_dimension)
        new_data = split_data(random_dimension,data,median)
        #print("L")
        create_kd_tree_aux(new_data[0], dimension_len, k, root)
        #print("R")
        create_kd_tree_aux(new_data[1], dimension_len, k, root) 

def split_data(dimension, data, root):
    new_data_left = []
    new_data_right = []
    for x in data:
        if x[dimension] < root:
            new_data_left.append(x)
        else:
            new_data_right.append(x)
    new_data = []
    new_data.append(new_data_left)
    new_data.append(new_data_right)
    return new_data

def test_tree(dataTest):
    data = [[7,8,9,4,5,6],[1,2,3,7,5,6],[5,7,8,3,1,9],[10,12,4,6,4,7],[4,9,4,7,1,3],[4,7,13,9,1,2]]
    tree = create_kd_tree(data,2)
    test = [7,2,3,4,1]
    y = find_neighbors(tree,test)
    test.append(y)
    print("RESULT {}".format(test))
    


def find_neighbors(root, data):
    if root is not None:
        value = root.data
        dimension = root.dimension
        if dimension==-1:
            return get_nearest_neighbors(value)
        else:
            if data[dimension] < value:
                return find_neighbors(root.left,data)
            else:
                return find_neighbors(root.right,data)
    else:
        return None

def get_nearest_neighbors(data):
    results = {}
    for neighbors in data:
        y = neighbors[-1]
        if y not in results:
            y_appearances = 0
            for neighbors_appearance in data:
                if neighbors_appearance[-1]==y:
                    y_appearances+=1
            results.setdefault(y,y_appearances)
    return get_nearest_neighbors_aux(results)

def get_nearest_neighbors_aux(results):
    elements = results.items()
    cant_appearances = 0
    result = 0
    for y, value in elements:
        if value > cant_appearances:
            cant_appearances = value
            result=y
    return result
        
    
                 
                    

    




    

