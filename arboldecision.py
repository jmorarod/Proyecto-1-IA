

#Con estos datos de entrenamiento, se pretende crear un árbol de decisión
conjunto_entrenamiento = [
    ['verde',    3, 'manzana'],
    ['amarillo', 3, 'manzana'],
    ['rojo',     1,  'uva'   ],
    ['rojo',     1,  'uva'   ],
    ['amarillo', 3,  'limón' ],
]

#Estos valores representan el valor de cada columna
encabezados = ['color', 'diámetro', 'valor']

#Esta función devuelve una lista, que consiste en los valores para la columna que se le pase como parámetro
def obtener_conjunto_columna(filas, columna):
    valores_columna = []
    numero_filas = len(filas)
    for i in range(numero_filas):
        valor_tomar = filas[i][columna]
        valores_columna.append(valor_tomar)
    return valores_columna


#Esta función recibe como parámetro el conjunto de entranamiento y una columna, de manera que se obtenga un conjunto con
#los valores únicos para esa columna
def valores_unicos_por_columna(filas, columna):
    valores_columna = obtener_conjunto_columna(filas, columna)
    conjunto_valores_columna = set(valores_columna)
    return conjunto_valores_columna
