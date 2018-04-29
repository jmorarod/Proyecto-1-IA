import math

# con estos datos de entrenamiento, se pretende crear un árbol de decisión

"""
conjunto_entrenamiento = [
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, 0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.2, -0.3, 0.3, 'NO', 0.3, 'SI', 'SI', -0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.1, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'NO', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, 0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [-0.9, 0.3, -0.3, 'NO', -0.3, 'SI', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 24', 'FRENTE AMPLIO'],
    [0.3, 0.3, -0.3, 'SI', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, 0.3, '20 a 50', 'RESTAURACION NACIONAL'],
    [0.3, 0.3, -0.3, 'NO', -0.3, 'SI', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'SI', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'NO', 'NO', 'NO', 0.3, -0.3, -0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [-0.3, -0.3, 0.3, 'NO', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'NO', 'SI', 'SI', 'SI', 0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, 0.3, '20 a 50', 'RESTAURACION NACIONAL'],
    [-0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, 0.3, -0.3, '20 a 30', 'ACCION CIUDADANA'],
    [0.3, 0.3, -0.3, 'SI', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', -0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 24', 'RESTAURACION NACIONAL'],
    [0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 50', 'ACCION CIUDADANA'],
    [0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, 0.3, '20 a 24', 'RESTAURACION NACIONAL'],
    [-0.3, -0.3, 0.3, 'NO', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, -0.3, '20 a 30', 'ACCION CIUDADANA'],
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, -0.3, '20 a 24', 'RESTAURACION NACIONAL'],
    [-0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [0.3, 0.3, -0.3, 'SI', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 50', 'RESTAURACION NACIONAL'],
    [0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, 0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [-0.3, -0.3, 0.3, 'NO', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, -0.3, '20 a 50', 'RESTAURACION NACIONAL'],
    [-0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, 0.3, '20 a 24', 'ACCION CIUDADANA'],
    [0.3, 0.3, -0.3, 'SI', -0.3, 'NO', 'NO', 0.3, 'CON VIVIENDA', 'CON VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, 0.3, -0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'SIN VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, -0.3, '20 a 60', 'RESTAURACION NACIONAL'],
    [-0.3, -0.3, 0.3, 'NO', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, 0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'CON VIVIENDA', 'CON VIVIENDA', 'SI', -0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, -0.3, -0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [-0.3, 0.3, -0.3, 'NO', -0.3, 'NO', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, 0.3, '20 a 24', 'ACCION CIUDADANA'],
    

]
"""

"""
# con estos datos de entrenamiento, se pretende crear un árbol de decisión
conjunto_entrenamiento = [
    ['young', 'myope', 'no', 'reduced', 'none'],
    ['young', 'myope', 'no', 'normal', 'soft'],
    ['young', 'myope', 'yes', 'reduced', 'none'],
    ['young', 'myope', 'yes', 'normal', 'hard'],
    ['young', 'hypermyope', 'no', 'reduced', 'none'],
    ['young', 'hypermyope', 'no', 'normal', 'soft'],
    ['young', 'hypermyope', 'yes', 'reduced', 'none'],
    ['young', 'hypermyope', 'yes', 'normal', 'hard'],
    ['pre-pre', 'myope', 'no', 'reduced', 'none'],
    ['pre-pre', 'myope', 'no', 'normal', 'soft'],
    ['pre-pre', 'myope', 'yes', 'reduced', 'none'],
    ['pre-pre', 'myope', 'yes', 'normal', 'hard'],
    ['pre-pre', 'hypermyope', 'no', 'reduced', 'none'],
    ['pre-pre', 'hypermyope', 'no', 'normal', 'soft'],
    ['pre-pre', 'hypermyope', 'yes', 'reduced', 'none'],
    ['pre-pre', 'hypermyope', 'yes', 'normal', 'none'],
    ['pre', 'myope', 'no', 'reduced', 'none'],
    ['pre', 'myope', 'no', 'normal', 'none'],
    ['pre', 'myope', 'yes', 'reduced', 'none'],
    ['pre', 'myope', 'yes', 'normal', 'hard'],
    ['pre', 'hypermyope', 'no', 'reduced', 'none'],
    ['pre', 'hypermyope', 'no', 'normal', 'soft'],
    ['pre', 'hypermyope', 'yes', 'reduced', 'none'],
    ['pre', 'hypermyope', 'yes', 'normal', 'none'],
    


]

"""
"""
conjunto_entrenamiento = [
    ['sunny', 'hot', 'high', 'false', 'no'],
    ['sunny', 'hot', 'high', 'true', 'no'],
    ['overcast', 'hot', 'high', 'false', 'yes'],
    ['rainy', 'mild', 'high', 'false', 'yes'],
    ['rainy', 'cool', 'normal', 'false', 'yes'],
    ['rainy', 'cool', 'normal', 'true', 'no'],
    ['overcast', 'cool', 'normal', 'true', 'yes'],
    ['sunny', 'mild', 'high', 'false', 'no'],
    ['sunny', 'cool', 'normal', 'false', 'yes'],
    ['rainy', 'mild', 'normal', 'false', 'yes'],
    ['sunny', 'mild', 'normal', 'true', 'yes'],
    ['overcast', 'mild', 'high', 'true', 'yes'],
    ['overcast', 'hot', 'normal', 'false', 'yes'],
    ['rainy', 'mild', 'high', 'true', 'no']

]
"""
c_entrenamiento = [['young', 'myope', 'no', 'normal', 'soft'], ['young', 'myope', 'yes', 'normal', 'hard'], ['young', 'hypermyope', 'no', 'normal', 'soft'], ['young', 'hypermyope', 'yes', 'normal', 'hard'], ['pre-pre', 'myope', 'no', 'normal', 'soft'], ['pre-pre', 'myope', 'yes', 'normal', 'hard'], ['pre-pre', 'hypermyope', 'no', 'normal', 'soft'], ['pre-pre', 'hypermyope', 'yes', 'normal', 'none'], ['pre', 'myope', 'no', 'normal', 'none'], ['pre', 'myope', 'yes', 'normal', 'hard'], ['pre', 'hypermyope', 'no', 'normal', 'soft'], ['pre', 'hypermyope', 'yes', 'normal', 'none']]

# estos valores representan el valor de cada columna
encabezados = ['color', 'diámetro', 'valor']

# esta función devuelve una lista, que consiste en los valores para la
# columna que se le pase como parámetro


def obtener_conjunto_columna(filas, columna):
    valores_columna = []
    numero_filas = len(filas)
    for i in range(numero_filas):
        valor_tomar = filas[i][columna]
        valores_columna.append(valor_tomar)
    return valores_columna


# esta función recibe como parámetro el conjunto de entranamiento y una columna, de manera que se obtenga un conjunto con
# los valores únicos para esa columna

def valores_unicos_por_columna(entrenamiento, columna):
    valores_columna = obtener_conjunto_columna(entrenamiento, columna)
    conjunto_valores_columna = set(valores_columna)
    return conjunto_valores_columna


# función que dada una lista con valores, retorna el índice en el que se
# encuentra el valor como segundo parámetro

def retornar_indice_valores(valores, valor):
    return valores.index(valor)

# función que recibe como parámetro el conjunto de entrenamiento, retorna
# la cantidad que hay de cada valor (última columna)

def contar_valores_conjunto_entrenamiento(conjunto_entrenamiento):
    valores = []
    cantidad_por_valor = []

    for i in range(len(conjunto_entrenamiento)):
        # tomando el último valor del encabezado
        valor = conjunto_entrenamiento[i][-1]
        if valor in valores:
            indice_valores = retornar_indice_valores(valores, valor)
            cantidad_por_valor[indice_valores] += 1
        else:
            valores.append(valor)
            cantidad_por_valor.append(1)
    return valores, cantidad_por_valor

#función que recibe como parámetro un valor, retornará si dicho valor es numérico o no

def es_numerico(valor):
    #isinstance es una función que puede evaluar el tipo de dato de una variable
    return isinstance(valor, int) or isinstance(valor, float)

#función encargada de obtener el tamaño de las columnas de los datos de entrenamiento

def obtener_tamano_columna_datos_entrenamiento(datos_entrenamiento):
    numero_columnas_retorno = len(datos_entrenamiento[0])
    for fila in datos_entrenamiento:
        if len(fila) != numero_columnas_retorno:
            return "error en el formato de los datos de entrenamiento, todas las filas deben tener la misma cantidad de columnas"
        else:
            numero_columnas_retorno = len(fila)
    return numero_columnas_retorno

#función encargada de obtener el tamaño de las filas de los datos de entrenamiento

def obtener_tamano_filas_datos_entrenamiento(datos_entrenamiento):
    return len(datos_entrenamiento)

#función que dado el valor de una columna, retorna todas las filas que contengan ese valor

def obtener_filas_por_valor_columna(datos_entrenamiento, valor_columna, columna):
    valor_retorno = []
    for fila in datos_entrenamiento:
        if valor_columna == fila[columna]:
            valor_retorno.append(fila)
    return valor_retorno

#función que, dado un conjunto, obtendrá las filas por categorías, dados los valores del conjunto

def obtener_filas_para_conjunto(datos_entrenamiento, conjunto, columna):
    filas = []
    for elemento in conjunto:
        lista = obtener_filas_por_valor_columna(datos_entrenamiento, elemento, columna)
        filas.append(lista)
        #contar_valores_conjunto_entrenamiento(lista)
    return filas


#XXXXfunción que separa los valores mayores a 0 y menores que 1 y los valores menores que 0 y mayores que -1

def obtener_filas_para_normalizado(datos_entrenamiento, columna):
    filas_menor_cero = []
    filas_mayor_cero = []
    for fila in datos_entrenamiento:
        if fila[columna] > 0 and fila[columna] <= 1:
            filas_mayor_cero.append(fila)
        else:
            filas_menor_cero.append(fila)
    contar_valores_conjunto_entrenamiento(filas_mayor_cero)
    


#función que determina si toda una columna es numérica o no

def es_columna_numerica(datos_entrenamiento, columna):
    for fila in datos_entrenamiento:
        if not es_numerico(fila[columna]):
            return False
    return True


#función encargada de obtener el valor de entropía, mediante la fórmula
def obtener_entropia_conjunto_entrenamiento_formula(probabilidades):
    resultado_entropia = 0
    for probabilidad in probabilidades:
        resultado_entropia -= probabilidad*math.log2(probabilidad)    
    return resultado_entropia
        


def obtener_entropia_conjunto_entrenamiento(entrenamiento):
    
    #obtiene la cantidad de votos para cada partido
    valores_etiquetas, votos_etiqueta = contar_valores_conjunto_entrenamiento(entrenamiento)
    #obtiene la cantidad de filas que tiene el conjunto de entrenamiento
    cantidad_filas = obtener_tamano_filas_datos_entrenamiento(entrenamiento)
    
    #se obtienen las probabilidades de voto para cada partido
    probabilidades_etiquetas = []
    tamano_votos_etiqueta = len(votos_etiqueta)
    for i in range(tamano_votos_etiqueta):
        probabilidad = votos_etiqueta[i] / cantidad_filas
        probabilidades_etiquetas.append(probabilidad)
    
    entropia = obtener_entropia_conjunto_entrenamiento_formula(probabilidades_etiquetas)
    return entropia




def obtener_probabilidades_fila(fila, largo_fila):
    resultado = []
    valores_etiquetas, votos_etiqueta = contar_valores_conjunto_entrenamiento(fila)
    for i in votos_etiqueta:
        resultado.append(i/largo_fila)
    return resultado

def resultado_logaritmo_probabilidad(probabilidades):
    resultado = 0
    for i in probabilidades:
        resultado-=i*math.log2(i)
    return resultado

    
def obtener_ganancia_columna(filas_conjunto, datos_entrenamiento, entropia):
    #se obtiene el número total de filas del conjunto de entrenamiento
    numero_filas = obtener_tamano_filas_datos_entrenamiento(datos_entrenamiento)
    resultado = 0
    for i in filas_conjunto:
        numero_filas_fila = len(i)
        probabilidad_fila = numero_filas_fila/numero_filas
        
        probabilidades = obtener_probabilidades_fila(i, numero_filas_fila)
        resultado_logaritmo = resultado_logaritmo_probabilidad(probabilidades)

        resultado+=probabilidad_fila*resultado_logaritmo

        #probabilidades_fila = obtener_probabilidades_fila
    ganancia = entropia-resultado
    return ganancia


#función encargada de recorrer las columnas de los datos de entrenamiento

def recorrer_columnas_datos_entrenamiento(datos_entrenamiento):
    print("DATOS_ENTRENAMIENTO")
    print(datos_entrenamiento)
    #primero, se obtiene la entropía
    entropia = obtener_entropia_conjunto_entrenamiento(datos_entrenamiento)
    #luego, se obtiene el número de columnas del conjunto de entrenamiento para recorrerlas
    numero_columnas = obtener_tamano_columna_datos_entrenamiento(datos_entrenamiento)
    #recorriendo las columnas del conjunto de entramiento

    ganancia_por_columna = []
    for i in range(numero_columnas-1):
        conjunto_fila_valores_diferentes = {}
        #se obtienen los valores diferentes que se pueden categorizar para la columna actual
        conjunto_fila_valores_diferentes = valores_unicos_por_columna(datos_entrenamiento, i)
        #se obtienen todas las filas que se relacionan con los valores categóricos obtenidos anteriormente
        filas_conjunto = obtener_filas_para_conjunto(datos_entrenamiento, conjunto_fila_valores_diferentes, i)
        #se obtiene la ganancia para la columna actual
        ganancia = obtener_ganancia_columna(filas_conjunto, datos_entrenamiento, entropia)
        ganancia_por_columna.append(ganancia)
    return ganancia_por_columna
        
       
    """
    numero_columnas = obtener_tamano_columna_datos_entrenamiento(datos_entrenamiento)
    for i in range(numero_columnas):
        conjunto_fila_valores_diferente = {}

        #si la columna es numérica, los valores se tomarán por rangos

        #if(es_columna_numerica(conjunto_entrenamiento, i)):
        #    obtener_filas_para_normalizado(conjunto_entrenamiento, i)
            
        #caso contrario, los valores se clasificarán por los posibles valores que tomará la columna

        #else:
        conjunto_fila_valores_diferente = valores_unicos_por_columna(datos_entrenamiento, i)
        print(conjunto_fila_valores_diferente)
        print("\n\n\n")
        obtener_filas_para_conjunto(conjunto_entrenamiento, conjunto_fila_valores_diferente, i)
    """


#función encargada de obtener el índice de la columna que genera más ganancia

def obtener_indice_maximo(ganancias):
    indice = 0
    maximo = ganancias[0]
    tamano_ganancias = len(ganancias)
    for i in range(1, tamano_ganancias):
        if ganancias[i]>maximo:
            maximo = ganancias[i]
            indice = i
    return indice

#función encargada de definir si un nodo es o no una hoja, de acuerdo a los target que obtiene

def es_nodo_hoja(valores, cantidad_por_valor):
    if len(valores) == 1 and len(cantidad_por_valor) == 1:
        return True
    else:
        return False

#función encargada de armar un árbol de decisión

def armar_arbol(conjunto_entrenamiento):

    #primero, se obtienen todas las ganancias por cada columna
    ganancias_por_columna = recorrer_columnas_datos_entrenamiento(conjunto_entrenamiento)
    
    print("GANANCIAS")
    print(ganancias_por_columna)

    #segundo, se obtiene la columna con más ganancia
    indice_maximo = obtener_indice_maximo(ganancias_por_columna)

    #tercero, se obtienen los valores diferentes que aporta la columna
    conjunto_fila_valores_diferentes = valores_unicos_por_columna(conjunto_entrenamiento, indice_maximo)
    
    #se recorre cada valor
    for i in conjunto_fila_valores_diferentes:
        #para cada valor, se obtienen las filas que cumplen con esta característica en la columna especificada
        filas_elemento = obtener_filas_por_valor_columna(conjunto_entrenamiento, i, indice_maximo)
       
        #se obtienen los target para dichas filas
        valores, cantidad_por_valor = contar_valores_conjunto_entrenamiento(filas_elemento)
        
        if not es_nodo_hoja(valores, cantidad_por_valor):
            print(obtener_entropia_conjunto_entrenamiento(filas_elemento))
            print(filas_elemento)
            print(recorrer_columnas_datos_entrenamiento(filas_elemento))
            #armar_arbol(filas_elemento)

        
    #print(ganancias_por_columna)

        


armar_arbol(c_entrenamiento)
#print(recorrer_columnas_datos_entrenamiento(conjunto_entrenamiento))

