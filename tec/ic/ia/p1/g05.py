#from tec.ic.ia.pc1.g05 import generar_muestra_pais, generar_muestra_provincia
from pc1 import generar_muestra_pais, generar_muestra_provincia
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers import SGD


def red_neuronal_r1_r2(muestra, num_capas, unidades_por_capa, activacion):
    x_train = []
    y_train = get_column(muestra,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 4)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=23))
    for i in range(1,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(4,activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128)
    return modelo

def red_neuronal_r2(muestra, num_capas, unidades_por_capa, activacion):
    x_train = []
    y_train = get_column(muestra,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 4)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=22))
    for i in range(1,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(4,activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128)
    return modelo

def red_neuronal_r1(muestra, num_capas, unidades_por_capa, activacion):
    x_train = []
    y_train = get_column(muestra,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 15)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=22))
    for i in range(1,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(15,activation='sigmoid'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128)
    return modelo
    
    
        
        

    
#Recibe la muestra generada por la funcion generar_muestra_pais o generar_muestra_provincia
def datos_r1_normalizados(muestra):
    targets = get_column(muestra,-2)
    x_vector = []
    for i in range (0, len(muestra)):
        x_vector += [muestra[i][:-2]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i,targets[j])]
        j += 1
    return return_list
    
def datos_r2_normalizados(muestra):
    targets = get_column(muestra,-1)
    x_vector = []
    for i in range (0, len(muestra)):
        x_vector += [muestra[i][:-2]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i,targets[j])]
        j += 1
    return return_list

def datos_r2_con_r1_normalizados(muestra):
    targets = get_column(muestra,-1)
    x_vector = []
    for i in range (0, len(muestra)):
        x_vector += [muestra[i][:-1]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i,targets[j])]
        j += 1
    return return_list
   


def get_column(matrix, i):
    return [row[i] for row in matrix]
