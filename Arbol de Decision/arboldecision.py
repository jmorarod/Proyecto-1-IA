import math
from arbol import Nodo, Hoja, Atributo
from g05 import datos_r1_normalizados, datos_r2_normalizados, datos_r2_con_r1_normalizados
from pc1 import generar_muestra_pais, generar_muestra_provincia
import numpy as np


# será útil para uno de los casos de los árboles de decisión, guardará los
# índice de los atributos de mayor utilidad
atributos_utilizados = []
encabezados = []
c_entrenamiento = []
c_pruebas = []
columnas_mayor_ocho = []



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

# función que recibe como parámetro un valor, retornará si dicho valor es
# numérico o no


def es_numerico(valor):
    # isinstance es una función que puede evaluar el tipo de dato de una
    # variable
    return isinstance(valor, int) or isinstance(valor, float)

# función encargada de obtener el tamaño de las columnas de los datos de
# entrenamiento


def obtener_tamano_columna_datos_entrenamiento(datos_entrenamiento):
    numero_columnas_retorno = len(datos_entrenamiento[0])
    for fila in datos_entrenamiento:
        if len(fila) != numero_columnas_retorno:
            return "error en el formato de los datos de entrenamiento, todas las filas deben tener la misma cantidad de columnas"
        else:
            numero_columnas_retorno = len(fila)
    return numero_columnas_retorno

# función encargada de obtener el tamaño de las filas de los datos de
# entrenamiento


def obtener_tamano_filas_datos_entrenamiento(datos_entrenamiento):
    return len(datos_entrenamiento)

# función que dado el valor de una columna, retorna todas las filas que
# contengan ese valor


def obtener_filas_por_valor_columna(
        datos_entrenamiento,
        valor_columna,
        columna):
    valor_retorno = []
    for fila in datos_entrenamiento:
        if valor_columna == fila[columna]:
            valor_retorno.append(fila)
    return valor_retorno

# función que, dado un conjunto, obtendrá las filas por categorías, dados
# los valores del conjunto


def obtener_filas_para_conjunto(datos_entrenamiento, conjunto, columna):
    filas = []
    for elemento in conjunto:
        lista = obtener_filas_por_valor_columna(
            datos_entrenamiento, elemento, columna)
        filas.append(lista)
        # contar_valores_conjunto_entrenamiento(lista)
    return filas


# función que determina si toda una columna es numérica o no

def es_columna_numerica(datos_entrenamiento, columna):
    for fila in datos_entrenamiento:
        if not es_numerico(fila[columna]):
            return False
    return True


# función encargada de obtener el valor de entropía, mediante la fórmula
def obtener_entropia_conjunto_entrenamiento_formula(probabilidades):
    resultado_entropia = 0
    for probabilidad in probabilidades:
        resultado_entropia -= probabilidad * math.log2(probabilidad)
    return resultado_entropia


def obtener_entropia_conjunto_entrenamiento(entrenamiento):

    # obtiene la cantidad de votos para cada partido
    valores_etiquetas, votos_etiqueta = contar_valores_conjunto_entrenamiento(
        entrenamiento)
    # obtiene la cantidad de filas que tiene el conjunto de entrenamiento
    cantidad_filas = obtener_tamano_filas_datos_entrenamiento(entrenamiento)

    # se obtienen las probabilidades de voto para cada partido
    probabilidades_etiquetas = []
    tamano_votos_etiqueta = len(votos_etiqueta)
    for i in range(tamano_votos_etiqueta):
        probabilidad = votos_etiqueta[i] / cantidad_filas
        probabilidades_etiquetas.append(probabilidad)

    entropia = obtener_entropia_conjunto_entrenamiento_formula(
        probabilidades_etiquetas)
    return entropia


def obtener_probabilidades_fila(fila, largo_fila):
    resultado = []
    valores_etiquetas, votos_etiqueta = contar_valores_conjunto_entrenamiento(
        fila)
    for i in votos_etiqueta:
        resultado.append(i / largo_fila)
    return resultado


def resultado_logaritmo_probabilidad(probabilidades):
    resultado = 0
    for i in probabilidades:
        resultado -= i * math.log2(i)
    return resultado


def obtener_ganancia_columna(filas_conjunto, datos_entrenamiento, entropia):
    # se obtiene el número total de filas del conjunto de entrenamiento
    numero_filas = obtener_tamano_filas_datos_entrenamiento(
        datos_entrenamiento)
    resultado = 0
    for i in filas_conjunto:
        numero_filas_fila = len(i)
        probabilidad_fila = numero_filas_fila / numero_filas

        probabilidades = obtener_probabilidades_fila(i, numero_filas_fila)
        resultado_logaritmo = resultado_logaritmo_probabilidad(probabilidades)

        resultado += probabilidad_fila * resultado_logaritmo

        #probabilidades_fila = obtener_probabilidades_fila
    ganancia = entropia - resultado
    return ganancia


# función encargada de recorrer las columnas de los datos de entrenamiento

def recorrer_columnas_datos_entrenamiento(datos_entrenamiento):
    # primero, se obtiene la entropía
    entropia = obtener_entropia_conjunto_entrenamiento(datos_entrenamiento)
    # luego, se obtiene el número de columnas del conjunto de entrenamiento
    # para recorrerlas
    numero_columnas = obtener_tamano_columna_datos_entrenamiento(
        datos_entrenamiento)
    # recorriendo las columnas del conjunto de entramiento

    ganancia_por_columna = []
    for i in range(numero_columnas - 1):
        conjunto_fila_valores_diferentes = {}
        # se obtienen los valores diferentes que se pueden categorizar para la
        # columna actual
        conjunto_fila_valores_diferentes = valores_unicos_por_columna(
            datos_entrenamiento, i)
        # se obtienen todas las filas que se relacionan con los valores
        # categóricos obtenidos anteriormente
        filas_conjunto = obtener_filas_para_conjunto(
            datos_entrenamiento, conjunto_fila_valores_diferentes, i)
        # se obtiene la ganancia para la columna actual
        ganancia = obtener_ganancia_columna(
            filas_conjunto, datos_entrenamiento, entropia)
        ganancia_por_columna.append(ganancia)
    return ganancia_por_columna

# función encargada de obtener el índice de la columna que genera más ganancia


def obtener_indice_maximo(ganancias):
    indice = 0
    maximo = ganancias[0]
    tamano_ganancias = len(ganancias)
    for i in range(1, tamano_ganancias):
        if ganancias[i] > maximo:
            maximo = ganancias[i]
            indice = i
    return indice

# función encargada de definir si un nodo es o no una hoja, de acuerdo a
# los target que obtiene


def es_nodo_hoja(valores, cantidad_por_valor):
    if len(valores) == 1 and len(cantidad_por_valor) == 1:
        return True
    else:
        return False

# función que indica si una lista de ganancias no aporta


def es_ganancia_cero(ganancias):
    for i in ganancias:
        if i != 0.0:
            return False
    return True

# función encargada de retornar el valor de un target


def retornar_target(filas):
    fila_actual = filas[0][-1]
    return fila_actual


def obtener_max_lista(valores, cantidad_por_valor):
    tamano = len(valores)
    maximo = cantidad_por_valor[0]
    indice_devolver = 0
    for i in range(1, tamano):
        if cantidad_por_valor[i] > maximo:
            maximo = cantidad_por_valor[i]
            indice_devolver = i
    return valores[indice_devolver]


def obtener_pluralidad(filas):
    valores = []
    cantidad_por_valor = []

    for i in range(len(filas)):
        # tomando el último valor del encabezado
        valor = filas[i][-1]
        if valor in valores:
            indice_valores = retornar_indice_valores(valores, valor)
            cantidad_por_valor[indice_valores] += 1
        else:
            valores.append(valor)
            cantidad_por_valor.append(1)
    return obtener_max_lista(valores, cantidad_por_valor)


def son_todos_target_iguales(filas):
    largo = len(filas)
    target = filas[0][-1]
    for i in range(1, largo):
        if filas[i][-1] != target:
            return False
    return True


def obtener_lista_atributos_utilizados():
    lista_retorno = []
    for i in atributos_utilizados:
        lista_retorno.append(i.nombre)
    return lista_retorno

# función encargada de devolver las ganancias que se pueden utilizar para


def reducir_ganancias_por_atributos_utilizados(ganancias):
    respuesta_ganancias = []
    respuesta_encabezados = []
    atributos = obtener_lista_atributos_utilizados()
    tamano = len(ganancias)
    for i in range(tamano):
        if encabezados[i] not in atributos:
            respuesta_ganancias.append(ganancias[i])
            respuesta_encabezados.append(encabezados[i])
    return respuesta_ganancias, respuesta_encabezados


def obtener_indice_encabezado(nombre):
    return encabezados.index(nombre)


def obtener_filas_mayores_menores_cero(conjunto, columna):
    mayores = []
    menores = []
    for i in conjunto:
        if i[columna] > 0:
            mayores.append(i)
        elif i[columna] < 0:
            menores.append(i)
    return mayores, menores


def armar_arbol(conjunto_entrenamiento, filas_padre):
    tipo_nodo = 0
    # caso 1, se pregunta si quedan ejemplos disponibles para este camino
    if conjunto_entrenamiento == []:
        target = obtener_pluralidad(filas_padre)
        hoja = Hoja(target)
        return hoja
    else:
        # caso 2, todos los target son iguales para el conjunto de
        # entrenamiento
        if(son_todos_target_iguales(conjunto_entrenamiento)):
            target = retornar_target(conjunto_entrenamiento)
            hoja = Hoja(target)
            return hoja

        else:

            # hacer split
            # 1.obtener las ganancias para cada columna
            ganancias_por_columna = recorrer_columnas_datos_entrenamiento(
                conjunto_entrenamiento)
            ganancias_permitidas, encabezados_permitidos = reducir_ganancias_por_atributos_utilizados(
                ganancias_por_columna)
            # 2.ver si quedan atributos disponibles para hacer split
            if ganancias_permitidas == []:
                #print("ya no hay atributos, pluralidad ejemplos")
                target = obtener_pluralidad(conjunto_entrenamiento)
                hoja = Hoja(target)
                return hoja
                # crear una hoja aquí
            else:
                hijos = []
                indice_maximo = obtener_indice_maximo(ganancias_permitidas)

                encabezado_nodo = encabezados_permitidos[indice_maximo]
                ganancia_nodo = ganancias_permitidas[indice_maximo]

                atributo = Atributo(ganancia_nodo, encabezado_nodo)

                indice_nodo = obtener_indice_encabezado(encabezado_nodo)

                atributos_utilizados.append(atributo)
                conjunto_fila_valores_diferentes = valores_unicos_por_columna(
                    conjunto_entrenamiento, indice_nodo)

                # si se aporta gran cantidad de valores para un atributo, se
                # evaluará si es mayor o menor a 0
                if indice_nodo in columnas_mayor_ocho:
                    mayores, menores = obtener_filas_mayores_menores_cero(
                        conjunto_entrenamiento, indice_nodo)
                    nodo_mayores = armar_arbol(mayores, conjunto_entrenamiento)
                    nodo_menores = armar_arbol(menores, conjunto_entrenamiento)
                    hijos.append(nodo_mayores)
                    hijos.append(nodo_menores)

                else:
                    tipo_nodo = 1
                    for i in conjunto_fila_valores_diferentes:
                        # para cada valor, se obtienen las filas que cumplen
                        # con esta característica en la columna especificada
                        filas_elemento = obtener_filas_por_valor_columna(
                            conjunto_entrenamiento, i, indice_nodo)
                        # se llama de nuevo a la función, con los elementos del
                        # nuevo camino y los elementos del padre
                        nodo = armar_arbol(
                            filas_elemento, conjunto_entrenamiento)
                        hijos.append(nodo)
                nodo = Nodo(
                    hijos,
                    indice_nodo,
                    conjunto_fila_valores_diferentes,
                    ganancias_permitidas[indice_maximo],
                    tipo_nodo,
                    conjunto_entrenamiento)
                return nodo


def generar_header_conjunto_entrenamiento(conjunto_entrenamiento):
    tamano = len(conjunto_entrenamiento[0])
    for i in range(tamano - 1):
        encabezado = "Atributo: " + str(i)
        encabezados.append(encabezado)
    return encabezados


def recorrer_arbol(arbol):
    if(isinstance(arbol, Nodo)):
        print(arbol.ganancia)
        if es_nodo_con_hojas(arbol):
            print("SHIIIII")
        for i in arbol.hijos:
            recorrer_arbol(i)
    elif(isinstance(arbol, Hoja)):
        print(arbol.target)


def imprimir_columna(datos, columna):
    lista = []
    for i in datos:
        lista.append(i[columna])
    return lista


def obtener_valor_porcentaje_pruebas(n):
    return int(round(n * 0.25))


def partir_datos_entrenamiento_prueba(
        datos,
        cantidad_entrenamiento,
        cantidad_prueba):
    datos_entrenamiento = []
    datos_prueba = []
    for i in range(cantidad_entrenamiento):
        datos_entrenamiento.append(datos[i])
    for i in range(
            cantidad_entrenamiento,
            cantidad_entrenamiento +
            cantidad_prueba):
        datos_prueba.append(datos[i])
    return datos_entrenamiento, datos_prueba


def generar_arbol(n):

    cantidad_datos_prueba = obtener_valor_porcentaje_pruebas(n)
    cantidad_datos_entrenamiento = n - cantidad_datos_prueba
    print(cantidad_datos_entrenamiento)
    print(cantidad_datos_prueba)
    muestra = generar_muestra_pais(n)
    #data = datos_r1_normalizados(muestra)
    #data = datos_r2_normalizados(muestra)
    data = datos_r2_con_r1_normalizados(muestra)
    data = np.array(data).tolist()
        

    datos_entrenamiento, datos_prueba = partir_datos_entrenamiento_prueba(
        data, cantidad_datos_entrenamiento, cantidad_datos_prueba)
    c_entrenamiento = datos_entrenamiento
    c_pruebas = datos_prueba

    # se agregan los índices de los a tributos que aportan más de 8 opciones
    tamano = len(data[0])

    for i in range(tamano - 1):
        datos = imprimir_columna(data, i)
        datos_conjunto = set(datos)
        if len(datos_conjunto) >= 8:
            columnas_mayor_ocho.append(i)

    generar_header_conjunto_entrenamiento(c_entrenamiento)
    arbol = armar_arbol(c_entrenamiento, c_entrenamiento)
    return arbol, c_pruebas


def obtener_indice_conjunto(conjunto, valor):
    retorno = 0
    for i in conjunto:
        if i == valor:
            return retorno
        else:
            retorno += 1


def predecir(c_pruebas, arbol):
    predicciones = []
    valores_reales = []
    for i in c_pruebas:
        prediccion = predecir_aux(i, arbol)
        valores_reales.append(i[-1])
        predicciones.append(prediccion)
    return predicciones, valores_reales


def predecir_aux(fila, arbol):
    if isinstance(arbol, Nodo):
        columna = arbol.columna
        valor = fila[columna]
        conjunto = arbol.valores_columna
        if arbol.tipo == 0:
            if valor > 0:
                return predecir_aux(fila, arbol.hijos[0])
            elif valor < 0:
                return predecir_aux(fila, arbol.hijos[1])
        elif arbol.tipo == 1:
            indice = obtener_indice_conjunto(conjunto, valor)
            return predecir_aux(fila, arbol.hijos[indice])
    elif isinstance(arbol, Hoja):
        return arbol.target


def obtener_precision(verdaderos_positivos, falsos_positivos):
    return (verdaderos_positivos / (verdaderos_positivos + falsos_positivos))


def obtener_verdaderos_falsos_positivos(predicciones, reales):
    tamano = len(predicciones)
    verdaderos_positivos = 0
    falsos_positivos = 0
    for i in range(tamano):
        if predicciones[i] == reales[i]:
            verdaderos_positivos += 1
        else:
            falsos_positivos += 1
    return verdaderos_positivos, falsos_positivos


def es_nodo_con_hojas(arbol):
    for i in arbol.hijos:
        if not isinstance(i, Hoja):
            return False
    return True


def podar_arbol(arbol, umbral):
    if isinstance(arbol, Nodo):
        if es_nodo_con_hojas(arbol):
            print("HOJAAAAAAAAAAS")
            print(arbol.ganancia)
            if arbol.ganancia < umbral:
                target = obtener_pluralidad(arbol.filas)
                hoja = Hoja(target)
                return hoja
            else:
                return arbol
        else:
            tamano = len(arbol.hijos)
            hijos = []
            for i in range(tamano):
                print("hijo original")
                print(arbol.hijos[i])
                hijo = podar_arbol(arbol.hijos[i], umbral)
                print("hijo nuevo")
                print(hijo)
                hijos.append(hijo)
            arbol.hijos = hijos
            return arbol
            """
            if es_nodo_con_hojas(arbol.hijos[i]):
                if arbol.hijos[i].ganancia < umbral:
                    target = obtener_pluralidad(arbol.hijos[i].filas)
                    hoja = Hoja(target)
                    arbol.hijos[i] = hoja
            else:
                podar_arbol(arbol.hijos[i], umbral)

            """
    elif isinstance(arbol, Hoja):
        return arbol

    #return arbol
    #elif(isinstance(arbol, Hoja)):
    #    print(arbol.target)

def funcion_principal():
    arbol, c_pruebas = generar_arbol(10000)
    predicciones, valores_reales = predecir(c_pruebas, arbol)
    verdaderos_positivos, falsos_positivos = obtener_verdaderos_falsos_positivos(
        predicciones, valores_reales)
    print(verdaderos_positivos)
    print(falsos_positivos)
    precision = obtener_precision(verdaderos_positivos, falsos_positivos)
    print(precision)
    arbol_podado = podar_arbol(arbol, 0.08)
    #arbol_podado2 = podar_arbol(arbol_podado, 0.08)
    
    #arbol_podado3 = podar_arbol(arbol_podado2, 0.08)
    
    #arbol_podado4 = podar_arbol(arbol_podado3, 0.08)
    recorrer_arbol(arbol)
    print("\n\n\n\n\n")
    recorrer_arbol(arbol_podado)
    #recorrer_arbol(arbol_podado)


funcion_principal()
