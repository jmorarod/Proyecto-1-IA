#from tec.ic.ia.pc1.g05 import generar_muestra_pais, generar_muestra_provincia
from pc1 import generar_muestra_pais, generar_muestra_provincia
import numpy as np
from sklearn.preprocessing import StandardScaler

# Recibe la muestra generada por la funcion generar_muestra_pais o
# generar_muestra_provincia


def datos_r1_normalizados(muestra):
    targets = get_column(muestra, -2)
    x_vector = []
    for i in range(0, len(muestra)):
        x_vector += [muestra[i][:-2]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i, targets[j])]
        j += 1
    return return_list


def datos_r2_normalizados(muestra):
    targets = get_column(muestra, -1)
    x_vector = []
    for i in range(0, len(muestra)):
        x_vector += [muestra[i][:-2]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i, targets[j])]
        j += 1
    return return_list


def datos_r2_con_r1_normalizados(muestra):
    targets = get_column(muestra, -1)
    x_vector = []
    for i in range(0, len(muestra)):
        x_vector += [muestra[i][:-1]]
    standardScaler = StandardScaler()
    standardScaler.fit(x_vector)
    x_normalizado = standardScaler.transform(x_vector)
    return_list = []
    j = 0
    for i in x_normalizado:
        return_list += [np.append(i, targets[j])]
        j += 1
    return return_list


def get_column(matrix, i):
    return [row[i] for row in matrix]
