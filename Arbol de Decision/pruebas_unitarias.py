from arboldecision import *

#este será el conjunto de entrenamiento de prueba para estas pruebas
c_entrenamiento = [
    [-0.3, -0.3, 0.3, '1', 0.3, '1', '1', -0.3, '1', '-1', '1', 0.3, 0.3, '-1', '1', '1', '1', '1', 0.3, 0.3, 0.3, '1.1', '2.0'],
    [-0.2, -0.3, 0.3, '1', 0.3, '1', '1', -0.3, '-1', '1', '1', 0.3, 0.3, '-1', '1', '-1', '1', '1', 0.3, 0.3, -0.3, '1.0', '2.0'],
    [-0.1, -0.3, 0.3, '1', 0.3, '1', '1', -0.3, '1', '-1', '-1', 0.3, 0.3, '-1', '1', '1', '-1', '1', 0.3, 0.3, 0.3, '0.9', '1.0'],
    [-0.9, 0.3, -0.3, '1', -0.3, '1', '1', 0.3, '-1', '1', '1', -0.3, -0.3, '-1', '1', '-1', '1', '1', -0.3, -0.3, -0.3, '1.1', '3.0'],
    [0.3, 0.3, -0.3, '1', -0.3, '1', '1', 0.3, '1', '-1', '1', -0.3, -0.3, '-1', '1', '1', '1', '1', -0.3, -0.3, 0.3, '1.0', '1.0'],
    [0.3, 0.3, -0.3, '1', -0.3, '1', '-1', 0.3, '-1', '-1', '1', -0.3, -0.3, '1', '1', '1', '1', '1', -0.3, -0.3, -0.3, '0.9', '2.0'],
    [0.3, -0.3, 0.3, '1', 0.3, '1', '1', -0.3, '1', '1', '-1', 0.3, 0.3, '-1', '1', '-1', '-1', '1', 0.3, -0.3, -0.3, '1.1', '1.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '-1', '1', '1', 0.3, 0.3, '-1', '1', '1', '-1', '1', 0.3, 0.3, -0.3, '1.0', '2.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '-1', '-1', -0.3, '1', '1', '1', 0.3, -0.3, '-1', '1', '1', '1', '1', 0.3, 0.3, 0.3, '0.9', '1.0'],
    [-0.3, 0.3, -0.3, '-1', -0.3, '-1', '1', 0.3, '-1', '-1', '1', -0.3, -0.3, '-1', '1', '-1', '-1', '1', -0.3, 0.3, -0.3, '1.1', '2.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '-1', '1', 0.3, '1', '-1', '-1', -0.3, 0.3, '-1', '1', '1', '1', '1', -0.3, -0.3, -0.3, '1.0', '1.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '1', '1', 0.3, '-1', '1', '-1', -0.3, 0.3, '-1', '1', '1', '1', '1', -0.3, -0.3, -0.3, '0.9', '2.0'],
    [0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '1', '-1', '-1', 0.3, 0.3, '-1', '1', '-1', '-1', '1', 0.3, -0.3, 0.3, '1.1', '1.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '1', '1', '1', 0.3, 0.3, '-1', '1', '-1', '1', '1', 0.3, -0.3, -0.3, '1.0', '2.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '1', '-1', -0.3, '1', '-1', '1', 0.3, -0.3, '-1', '1', '-1', '1', '1', 0.3, 0.3, -0.3, '0.9', '1.0'],
    [-0.3, 0.3, -0.3, '-1', -0.3, '1', '-1', 0.3, '-1', '1', '1', -0.3, -0.3, '-1', '-1', '1', '-1', '1', -0.3, 0.3, -0.3, '1.1', '2.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '-1', '-1', 0.3, '-1', '-1', '1', -0.3, -0.3, '-1', '1', '1', '1', '1', -0.3, -0.3, -0.3, '1.0', '1.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '-1', '-1', 0.3, '1', '-1', '-1', -0.3, 0.3, '-1', '-1', '1', '-1', '1', -0.3, 0.3, -0.3, '0.9', '2.0'],
    [0.3, -0.3, 0.3, '-1', 0.3, '-1', '-1', -0.3, '1', '1', '1', 0.3, 0.3, '-1', '1', '1', '1', '1', 0.3, -0.3, 0.3, '1.1', '1.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '-1', '1', -0.3, '1', '-1', '1', 0.3, 0.3, '-1', '-1', '1', '1', '-1', 0.3, -0.3, -0.3, '1.0', '2.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '-1', '1', -0.3, '-1', '1', '1', 0.3, 0.3, '-1', '1', '-1', '-1', '1', 0.3, -0.3, -0.3, '0.9', '1.0'],
    [-0.3, 0.3, -0.3, '-1', -0.3, '-1', '1', 0.3, '1', '1', '-1', -0.3, -0.3, '-1', '-1', '-1', '1', '1', -0.3, -0.3, 0.3, '1.1', '2.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '1', '1', 0.3, '1', '-1', '-1', -0.3, -0.3, '-1', '1', '1', '1', '1', -0.3, 0.3, -0.3, '1.0', '1.0'],
    [0.3, 0.3, -0.3, '-1', -0.3, '1', '1', 0.3, '-1', '1', '-1', 0.3, 0.3, '-1', '-1', '1', '-1', '1', -0.3, 0.3, -0.3, '0.9', '2.0'],
    [0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '1', '-1', '1', 0.3, 0.3, '-1', '1', '1', '1', '1', 0.3, -0.3, -0.3, '1.0', '1.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '1', '1', '1', 0.3, -0.3, '-1', '-1', '-1', '-1', '1', 0.3, -0.3, 0.3, '1.1', '2.0'],
    [-0.3, -0.3, 0.3, '-1', 0.3, '1', '1', -0.3, '1', '-1', '1', -0.3, 0.3, '-1', '1', '-1', '1', '1', 0.3, -0.3, -0.3, '1.0', '1.0'],
    [-0.3, 0.3, -0.3, '-1', -0.3, '1', '-1', 0.3, '-1', '1', '1', -0.3, -0.3, '-1', '-1', '1', '-1', '1', -0.3, -0.3, 0.3, '0.1', '2.0'],
    

]

#prueba 1: comprobar que una columna específica del conjunto de datos de la lista correcta
def test_arbol_decision_1():
    lista = obtener_conjunto_columna(c_entrenamiento, 2)
    assert lista == [0.3, 0.3, 0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3, -0.3, -0.3, -0.3, 0.3, 0.3, 0.3, -0.3]

#comprobar que el árbol de decisión sea una referencia al nodo raiz, para la primera ronda
def test_arbol_decision_2():
    muestra = generar_muestra_pais(1000)
    data_r1 = datos_r1_normalizados(muestra)
    arbol_r1, c_pruebas_r1, c_entrenamiento_r1 = generar_arbol(
        1000, 25, data_r1)
    assert isinstance(arbol_r1, Nodo)

#comprobar que el árbol de decisión sea una referencia al nodo raiz, para la segunda ronda
def test_arbol_decision_3():
    limpiar_variables_globales()
    muestra = generar_muestra_pais(1000)
    data_r2 = datos_r2_normalizados(muestra)
    arbol_r2, c_pruebas_r2, c_entrenamiento_r2 = generar_arbol(
        1000, 25, data_r2)
    assert isinstance(arbol_r2, Nodo)

#comprobar que el árbol de decisión sea una referencia al nodo raiz, para la segunda + primera ronda
def test_arbol_decision_4():
    limpiar_variables_globales()
    muestra = generar_muestra_pais(1000)
    data_r2_r1 = datos_r2_con_r1_normalizados(muestra)
    arbol_r2_r1, c_pruebas_r2_r1, c_entrenamiento_r2_r1 = generar_arbol(
        1000, 25, data_r2_r1)
    assert isinstance(arbol_r2_r1, Nodo)

#comprobar que para el conjunto de datos dado, la cantidad de votos sea la correcta para cada partido
def test_arbol_decision_5():
    partidos, votos_por_partido = contar_valores_conjunto_entrenamiento(c_entrenamiento)
    assert partidos == ['2.0', '1.0', '3.0'] 
    assert votos_por_partido == [14, 13, 1]

#comprobar que la pluraridad sea la correcta
def test_arbol_decision_6():
    partido = obtener_pluralidad(c_entrenamiento)
    assert partido == '2.0'

#comprobar que la poda de un arbol retorna un nodo para la primera ronda
def test_arbol_decision_7():
    limpiar_variables_globales()
    muestra = generar_muestra_pais(1000)
    data_r1 = datos_r1_normalizados(muestra)
    arbol_r1, c_pruebas_r1, c_entrenamiento_r1 = generar_arbol(
        1000, 25, data_r1)
    arbol_podado = podar_arbol_aux_aux(arbol_r1, 0.08)
    assert isinstance(arbol_r1, Nodo)

#comprobar que la poda de un arbol retorna un nodo para la segunda ronda
def test_arbol_decision_8():
    limpiar_variables_globales()
    muestra = generar_muestra_pais(1000)
    data_r2 = datos_r2_normalizados(muestra)
    arbol_r2, c_pruebas_r2, c_entrenamiento_r2 = generar_arbol(
        1000, 25, data_r2)
    arbol_podado = podar_arbol_aux_aux(arbol_r2, 0.08)
    assert isinstance(arbol_r2, Nodo)

#comprobar que la poda de un arbol retorna un nodo para la segunda + primera ronda
def test_arbol_decision_9():
    limpiar_variables_globales()
    muestra = generar_muestra_pais(1000)
    data_r2_r1 = datos_r2_con_r1_normalizados(muestra)
    arbol_r2_r1, c_pruebas_r2_r1, c_entrenamiento_r2_r1 = generar_arbol(
        1000, 25, data_r2_r1)
    arbol_podado = podar_arbol_aux_aux(arbol_r2_r1, 0.08)
    assert isinstance(arbol_r2_r1, Nodo)


