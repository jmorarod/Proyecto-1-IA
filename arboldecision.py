

# con estos datos de entrenamiento, se pretende crear un árbol de decisión
conjunto_entrenamiento = [
    [-0.3, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, 0.3, '20 a 24', 'RESTAURACION NACIONAL'],
    [-0.2, -0.3, 0.3, 'NO', 0.3, 'SI', 'SI', -0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
    [-0.1, -0.3, 0.3, 'SI', 0.3, 'NO', 'NO', -0.3, 'SIN VIVIENDA', 'CON VIVIENDA', 'NO', 0.3, 0.3, 'NO', 'SI', 'SI', 'SI', 'SI', 0.3, 0.3, 0.3, '20 a 30', 'RESTAURACION NACIONAL'],
    [-0.9, 0.3, -0.3, 'NO', -0.3, 'SI', 'NO', 0.3, 'CON VIVIENDA', 'SIN VIVIENDA', 'SI', -0.3, -0.3, 'NO', 'SI', 'SI', 'SI', 'SI', -0.3, -0.3, -0.3, '20 a 24', 'ACCION CIUDADANA'],
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
    print(valores)
    print(cantidad_por_valor)

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
    for elemento in conjunto:
        lista = obtener_filas_por_valor_columna(datos_entrenamiento, elemento, columna)
        print("PARA ELEMENTO: " + str(elemento))
        print(lista)
        print("\n\n\n")

def obtener_filas_para_normalizado(datos_entrenamiento, columna):
    filas_menor_cero = []
    filas_mayor_cero = []
    print("NORMALIZADO")
    for fila in datos_entrenamiento:
        if fila[columna] > 0 and fila[columna] <= 1:
            filas_mayor_cero.append(fila)
        else:
            filas_menor_cero.append(fila)
    print(filas_mayor_cero)
    print("\n")
    print(filas_menor_cero)
    print("\n\n\n")




def es_columna_numerica(datos_entrenamiento, columna):
    for fila in datos_entrenamiento:
        if not es_numerico(fila[columna]):
            return False
    return True

#función encargada de recorrer las columnas de los datos de entrenamiento

def recorrer_columnas_datos_entrenamiento(datos_entrenamiento):
    numero_columnas = obtener_tamano_columna_datos_entrenamiento(datos_entrenamiento)
    for i in range(numero_columnas):
        conjunto_fila_valores_diferente = {}

        #si la columna es numérica, los valores se tomarán por rangos

        if(es_columna_numerica(conjunto_entrenamiento, i)):
            obtener_filas_para_normalizado(conjunto_entrenamiento, i)
            
        #caso contrario, los valores se clasificarán por los posibles valores que tomará la columna

        else:
            conjunto_fila_valores_diferente = valores_unicos_por_columna(datos_entrenamiento, i)
            obtener_filas_para_conjunto(conjunto_entrenamiento, conjunto_fila_valores_diferente, i)
        

        


recorrer_columnas_datos_entrenamiento(conjunto_entrenamiento)

