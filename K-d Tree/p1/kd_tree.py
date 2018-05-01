import math
import random
import numpy as np

from g05 import datos_r1_normalizados,datos_r2_normalizados,datos_r2_con_r1_normalizados
from pc1 import generar_muestra_pais, generar_muestra_provincia

class Node:

    def __init__(self, dimension, data):
        self.left = None
        self.right = None
        self.dimension = dimension 
        self.data = data
    
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
    if k < (len(data)//100):
        k = (len(data)//100)
        print ("\nEl k es muy pequeño para la cantidad de datos,"
               " se utilizara el siguiente k : {} \n".format(k))
    return create_kd_tree_aux(data, dimension_len, k, None, 0)
    

    

def create_kd_tree_aux(data, dimension_len, k, root,level):
    median=0
    random_dimension = level % (dimension_len-1)
    if root is None:
        while(median==0):
            for x in data:
                median += x[random_dimension]
            if median==0:
                level += 1
                random_dimension = level % (dimension_len-1)
        median = median / len(data)
        new_root = Node(random_dimension, median)
        new_data = split_data(random_dimension,data,median)
        create_kd_tree_aux(new_data[0], dimension_len, k, new_root, level+1)
        create_kd_tree_aux(new_data[1], dimension_len, k, new_root, level+1)
        return new_root;
    elif len(data)<=k:
        if len(data)!=0:
            if root.data > data[0][root.dimension]:
                root.insert_leaf(data,True)
            else:
                root.insert_leaf(data,False)
        else:
            root.insert_leaf(None,True)
            root.insert_leaf(None,False)
    else:
        while(median==0):
            for x in data:
                median += x[random_dimension]
            if median==0:
                level += 1
                random_dimension = level % (dimension_len-1)
        median = median / len(data)
        new_root=root.insert_node(median,random_dimension)
        new_data = split_data(random_dimension,data,median)
        create_kd_tree_aux(new_data[0], dimension_len, k, new_root, level+1)
        create_kd_tree_aux(new_data[1], dimension_len, k, new_root, level+1) 

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
        left = root.left
        right = root.right
        if value!=None:
            if dimension==-1:
                return get_nearest_neighbors(value)
            else:
                if data[dimension] < value and left!=None:
                    return find_neighbors(left,data)
                elif data[dimension] >= value and right!=None:
                    return find_neighbors(right,data)
                else:
                    return None
        else:
            return None
    else:
        return None

def get_nearest_neighbors(data):
    results = {}
    if (len(data)>1):
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

def training(n, k, percentage):
    correct = 0
    incorrect = 0
    if(percentage<=100 and percentage>0):
        muestra = generar_muestra_pais(n)
        #data = datos_r1_normalizados(muestra)
        #data = datos_r2_normalizados(muestra)
        data = datos_r2_con_r1_normalizados(muestra)
        data = np.array(data).tolist()
        percentage = n * (percentage/100)
        percentage = int(round(percentage,0))
        data_training = data[:percentage]
        data_testing = data[percentage:]
        try:
            tree =create_kd_tree(data_training,k)
        except RecursionError as re:
            print("No se pudo resolver la creacion del arbol, puede deberse al tamaño de k. ERROR:"
                  " {} ".format(re.args[0]))
            return None
            
        for data_test in data_testing:
            y = find_neighbors(tree, data_test)
            data_test.append(y)
            if(data_test[-1]==data_test[-2]):
                correct += 1
            else:
                incorrect += 1
        print("\n El porcetaje de aciertos es de {}%".format((correct/len(data_testing))*100))
    else:
        print("El valor del porcentaje es incorrecto utilizar un valor entre 1 - 100")
                    

    




    

