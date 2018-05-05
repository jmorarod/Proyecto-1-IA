
# Informe de proyecto #1 - Inteligencia Artificial: Predicción de Votaciones
#### Realizado Por:
1. José Miguel Mora Rodríguez
2. Dylan Rodríguez Barboza
3. Karina Zeledón Pinell
#### Profesor:
Juan Manuel Esquivel Rodríguez, Ph. D.

# Simulación de votantes
Como base para el proyecto se tomó el módulo programado como proyecto corto #1 del curso de Inteligencia Artificial el cuál consistía en la generación de votantes(podían ser divididos por provincia o no) de diferentes cantones del país, generando ciertos indicadores y su voto en la primera ronda de elecciones del 2018. Se realizó una modificación para generar de igual forma el voto de cada uno en la segunda ronda de las elecciones.
## Indicadores cantonales

Se tomaron datos del censo del 2011 con los cuáles se generan los siguientes datos:

|Número de atributo|Atributo|Forma en que se genera
|------------------|--------|----------------------	
0|Población total|Se toma el mismo del canton
1|Superficie (km2)|Se toma el mismo del canton
2|Densidad de población|Se toma el mismo del canton
3|Personas por km2|Se toma el mismo del canton
4|Población urbana|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
5|Sexo|Se realiza una tabla de probabilidad acumulada entre HOMBRE o MUJER y se realiza la muestra
6|Relación de dependencia demográfica|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
7|Ocupa vivienda individual|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
8|Promedio de ocupantes|Se toma el mismo del canton
9|Vivienda en buen estado|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
10|Vivienda hacinada|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
11|Alfabetismo|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
12|Escolaridad Promedio|Se toma el mismo del canton
13|Asistencia a educación regular|Se toma el mismo del canton
14|Fuera de la fuerza de trabajo|Depende del atributo 15
15|En la fuerza de trabajo|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
16|Porcentaje de población ocupada no asegurada|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
17|Nacido en el extranjero|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
18|Discapacidad|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
19|No asegurado|Se realiza una tabla de probabilidad acumulada entre SI o NO y se realiza la muestra
20|Porcentaje de hogares con jefatura femenina|Se toma el mismo del canton
21|Porcentaje de hogares con jefatura compartida|Se toma el mismo del canton
22|Edad|Dado el sexo se realiza una muestra por rangos de edad
23|Voto en primera ronda|Se realiza una tabla de probabilidad acumulada por opcion y se realiza la muestra
24|Voto en segunda ronda|Se realiza una tabla de probabilidad acumulada por opcion y se realiza la muestra

Las probabilidades acumuladas de las edades se toman de los datos del censo los cuáles se presentan en el mismo de la siguiente forma:
![Distribucion de edades](http://res.cloudinary.com/chaldeacloud/image/upload/v1525494782/Proyecto%20IA/Ejemplo_distribuci%C3%B3n_edades.png)
De los datos anteriores se descartan los rangos de "0 a 4" hasta "10 a 14", además, se toma la suposición de que cualquier votante generado en el rango de 15 a 19, tiene al menos 18 años. Para generar la tabla de probabilidad acumulada se toma el total de personas en cada rango y se divide entre el total. Además el simulador toma en cuenta la distinción por distribución de edades entre hombres y mujeres por lo que si se genera un votante hombre se realizará una muestra en los datos correspondientes a la distribución de edades de hombres y de la misma forma si fuera una mujer la votante generada.

Los actas de escrutinio de votos se encuentran mapeadas por junta receptora, por lo que previamiente se realizó una asociación de las juntas con su cantón respectivo de forma que los votos por cada opción se encontraran relacionados con el cantón y así generar una tabla de probabilidad acumulada, con la cuál se genera un número aleatorio entre 0 y 1 y la opción de voto generada es la que cuya probabilidad acumulada sea mayor al número aleatorio y el número aleatorio sea mayor a la opción de voto anterior en la tabla, por ejemplo: se generó el número 0.65.

Opcion de voto|Probabilidad|Probabilidad Acumulada
--------------|------------|---------------------|
Opcion 1| 0.6|0.6
Opcion 2|0.1|0.7
Opcion3 |0.3|1
*La probabilidad de cada opción se calcula con los votos que obtuvo la respectiva opción dividido entre la totalidad de los votos.
Como el número generado es menor a la probabilidad acumulada de la opción 2 (se revisa primero esta opción antes que la opción 3) entonces el voto generado será de la opción 3. Los procesos de muestreo mencionados con anterioridad se realizan de la misma forma.

# Análisis de Resultados
En esta sección se mostrará el análisis de los resultados obtenidos de la aplicación de distintos modelos de aprendizaje automático para la predicción de los votos en ambas rondas electorales en Costa Rica.
## Regresión Logísitica
Para la predicción de los votos de los votantes generados en el simulador descrito anteriormente se implementó
un método de clasificación utilizando regresión logística mediante el uso de Tensorflow, el cuál dados los indicadores de los votantes predice en el caso de la primera ronda por cuál de los 13 partidos votó el votante generado o bien si su voto fué nulo o blanco y en el caso de la segunda ronda por cuál de los dos partidos fué su voto o bien si el mismo fué nulo o blanco.
Como parte del análisis de resultados del método de regresión logística es importante analizar los siguientes aspectos:
* Se utiliza la función softmax como función para la clasificacion, la cual tiene la siguiente forma:
![Funcion Softmax](https://qph.fs.quoracdn.net/main-qimg-fda2f008df90ed5d7b6aff89b881e1ac)
* Se utilizan los siguientes parámetros:
  * Learning rate: Influye en que tan rápido converge el modelo y que tan preciso es(en la tabla a continuación se muestran pruebas con diferentes valores)
  * Regularización: Forma en la que se evita que el modelo tenga problemas de overfitting(overfitting quiere decir que el modelo se cesgue a los datos de entrenamiento, evitando una buena generalización).
	* L1: Utiliza el valor absoluto de la resta del valor real menos el predicho  Tiende a crear modelos sparse, lo que quiere decir que sus parámetros(los pesos W que acompañan a las variables, ejemplo: x1 * w1 + w0, en este caso w1 y w0 pertenecen a W).
	* L2: Utiliza la resta del valor real menos el predicho al cuadrado

Utilizando un 80% del conjunto de datos como muestras de entrenamiento se obtuvieron los siguientes resultado:
 
Predicción |Tamaño de muestra| Learning Rate | Regularizacion | Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-----------|---------------|---------------|----------------|------------------------|----------------------------|------------------|--------------------|
Ronda 1 | 10000 | 0.1 | L1 | 30519141.7126 | 0.257625 | 7579275.5722 | 0.3025
Ronda 1 | 10000 | 0.5 | L1 | 30251626.8525 | 0.281125 | 7510920.3164 | 0.319
Ronda 1 | 10000 | 1 | L1 | 30237538.3920 | 0.283 | 7504048.3459 | 0.3215
Ronda 1 | 10000 | 10 | L1 | 30503154.7712 | 0.27025 | 7632836.1262 | 0.273
Ronda 1 | 10000 | 0.1 | L2 | 30518151.0483 | 0.2596 | 7582307.8200 | 0.298
Ronda 1 | 10000 | 0.5 | L2 | 30250394.9992 | 0.281125 | 7511138.6596 | 0.319
Ronda 1 | 10000 | 1 | L2 | 30253751.3774 | 0.282125 | 7503252.0053 | 0.323
Ronda 1 | 10000 | 10 | L2 | 30344709.9606 | 0.2825 | 7548750.2258 | 0.3035
Ronda 2 | 10000 | 0.1 | L1 | 9491.9661 | 0.544625 | 3576598.6481 |  0.5445
Ronda 2 | 10000 | 0.5 | L1 | 9060.5018 | 0.590125 | 3336570.5162 |  0.6225
Ronda 2 | 10000 | 1 | L1 | 9060.5018 | 0.5885 | 3335556.8941 |   0.624
Ronda 2 | 10000 | 10 | L1 | 9453.4559 | 0.5545 | 3578278.3964 |  0.554
Ronda 2 | 10000 | 0.1 | L2 | 9491.9661 | 0.544625 | 3576598.6481 |  0.5445
Ronda 2 | 10000 | 0.5 | L2 | 9060.5018 |  0.590125 | 3336570.5162 |  0.6225
Ronda 2 | 10000 | 1 | L2 | 9058.8274 |  0.5885 | 3335556.8941 |   0.624
Ronda 2 | 10000 | 10 | L2 | 9453.4559 | 0.5545 | 3578278.3964 |  0.554
Ronda 2 con Ronda 1 | 10000 | 0.1 | L1 |  9047.1585 | 0.591 | 3325170.7272 |  0.635
Ronda 2 con Ronda 1 | 10000 | 0.5 | L1 | 9035.0212 | 0.589875 | 3318207.01525 |  0.636
Ronda 2 con Ronda 1 | 10000 | 1 | L1 | 9034.5515 | 0.588375 | 3317706.0003 | 0.6375
Ronda 2 con Ronda 1 | 10000 | 10 | L1 | 9440.9166 | 0.55525 | 3505429.7438 |  0.5765
Ronda 2 con Ronda 1 | 10000 | 0.1 | L2 |   9047.15852 | 0.591 | 3325170.7272 |  0.635
Ronda 2 con Ronda 1 | 10000 | 0.5 | L2 | 9035.0212 | 0.589875 | 3318207.01525 |  0.636
Ronda 2 con Ronda 1 | 10000 | 1 | L2 | 9034.5515 | 0.588375 | 3317706.0003 | 0.6375
Ronda 2 con Ronda 1 | 10000 | 10 | L2 | 9440.9166 | 0.55525 | 3505429.7438 |  0.5765

Los datos presentados anteriormente se pueden visualizar de forma gráfica en los siguientes gráficos, el eje y representa la precisión y el eje x el learning rate utilizado:
![Entrenamiento R1 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525469802/Entrenamiento_R1_Regresion.png)
![Pruebas R1 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525469802/Test_R1_Regresion.png)
![Entrenamiento R2 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525470019/Proyecto%20IA/Entrenamiento_R2_Regresion.png)
![Pruebas R2 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525470019/Proyecto%20IA/Test_R2_Regresion.png)
![Entrenamiento R2_R1 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525470090/Proyecto%20IA/Entrenamiento_R2_con_R1_Regresion.png)
![Pruebas R2_R1 Regresion](http://res.cloudinary.com/chaldeacloud/image/upload/v1525470090/Proyecto%20IA/Test_R2_con_R1_Regresion.png)

Dados los resultados anteriores se puede concluir lo siguiente:

* La predicción de la primera ronda tiende a dar valores de precisión bajos debido a la cantidad de posibles salidas(15 opciones de voto)
* La regularización L1 o L2 muestra mayor influencia cuando hay mas posibles opciones a predecir(es decir en la primera ronda de votaciones)
* Con muestras mayores a 10000 en potencias de 10 ^ n se obtuvo que el mejor learning rate tiende a ser (5000 / número de muestras)
* En el mejor de los casos el modelo tiene una precisión de entre 62% y 63%, lo cuál se debe a varios aspectos:
	*La naturaleza de los datos(son simulados).
	*Algunos datos los cuáles se creen ayudarían a la generalización no fueron muestreados ya que eran promedios, por ejemplo: promedio de escolaridad.
	*No se conoce la distribución verdadera de cada columna por lo cuál se opta por normalizar con media de 0 y desviación de 1 todos los datos.

##Redes Neuronales
Otro método utilizado para la predicción de los votos es el de las redes neuronales, el cuál consiste en la conexión de varias capas de nodos para a raíz de una entrada generar una o varias salidas. Se puede apreciar la estructura de una red neuronal en la siguiente imágen:
![Ejemplo Red Neuronal](http://robologs.net/wp-content/uploads/2017/01/ejemplo_2_capas_ocultas.png)
En el caso de este proyecto la cantidad de entradas es la misma que la cantidad de indicadores o si se quiere predecir los votos de segunda ronda dados los votos de la primera, entonces sería la cantidad de indicadores + 1. La capa de salida en el caso de la predicción de la primera ronda contiene 15 nodos, uno por cada opción de partido; en los otros dos casos cuenta con 4 nodos de salida.
Para el análisis de los resultados obtenidos se evaluaron tres estructuras diferentes(cantidad de capas entre la entrada y salida y cantidad de nodos) con dos funciones de activación diferentes(de ellas depende si un nodo envía una señal al siguiente nodo). Las funciones de activación las cuáles pueden ser enviadas al programa mediante la bandera --funcion-activacion son las siguientes:
* 'relu'
* 'softmax'
* 'sigmoid'
* 'tanh'
* 'softplus'
* 'linear'
* 'hard_sigmoid'
* etc. (todas las funciones aceptadas por keras)

Es importante destacar que la capa de entrada siempre cuenta con la función 'relu' y la de salida con 'softmax'. Para el análisis se evaluarán las funciones 'relu' y 'softmax' con las siguientes estructuras:
* 3 capas escondidas de 30 neuronas y 30 neuronas en la capa de entrada.
* 4 capas escondidas de 30, 40, 30 y 35 neuronas respectivamente.
* 2 capas escondidas de 50 y 40 neuronas.

Los siguientes son los resultados de la primera estructura con la función 'relu' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.8982 | 0.28975 |  1.9106 | 0.271
Ronda 2 | 1 | 0.71998 | 0.612875 | 0.7333 | 0.5955
Ronda 2 con Ronda 1 | 1 |  0.7181 | 0.61025 | 0.73517 | 0.5895
Ronda 1 | 2 | 1.8896 | 0.290375 |  1.9121 | 0.2650
Ronda 2 | 2 | 0.7028 | 0.613875 | 0.7351 | 0.5944
Ronda 2 con Ronda 1 | 2 | 0.6966 |0.62475 | 0.73867 | 0.591
Ronda 1 | 3 | 1.8803 | 0.30125 |  1.9172 | 0.2889
Ronda 2 | 3 | 0.7148 | 0.6135 |0.7197 | 0.6094
Ronda 2 con Ronda 1 | 3 |  0.7141 |0.6142|0.7222| 0.5960
Ronda 1 | 4 | 1.8841 | 0.2845 |  1.9353 | 0.2749
Ronda 2 | 4 | 0.7102 | 0.6052| 0.7012 | 0.6084
Ronda 2 con Ronda 1 | 4 |  0.7093 |0.6067|0.7080| 0.6030
Ronda 1 | 5 | 1.8904 | 0.2906 | 1.9046 | 0.2889
Ronda 2 | 5 | 0.7091 | 0.6106| 0.7318 | 0.5905
Ronda 2 con Ronda 1 | 5 |   0.7132 | 0.602125 | 0.7388| 0.6000
Utilizando el promedio de los datos tabulados se obtiene la siguiente gráfica:
![Precision RN 1 Relu](http://res.cloudinary.com/chaldeacloud/image/upload/v1525473898/Proyecto%20IA/Precision_RN_1_Relu.png)

Los siguientes son los resultados de la primera estructura con la función 'softmax' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.9260 | 0.255 |  1.9135 | 0.269
Ronda 2 | 1 | 0.7403 | 0.5828 |  0.7204 | 0.6010
Ronda 2 con ronda 1 | 1 | 0.7403 | 0.5828 |  0.7202 | 0.6010
Ronda 1 | 2 | 1.9237 | 0.2578 |  1.9172 | 0.2519
Ronda 2 | 2 | 0.7377 |0.5796 |  0.7356 | 0.5800
Ronda 2 con ronda 1 | 2 | 0.7377 | 0.5796 |  0.7355 |  0.5800
Ronda 1 | 3 | 1.9207 | 0.2623 |  1.9640 | 0.253
Ronda 2 | 3 | 0.7410 |0.5803 |  0.7457 | 0.5864
Ronda 2 con ronda 1 | 3 | 0.7409 | 0.5803 |  0.7458 |  0.5864
Ronda 1 | 4 | 1.9331 | 0.2531 |  1.9164 | 0.2545
Ronda 2 | 4 | 0.7353 | 0.5837 |  0.7137 | 0.5919
Ronda 2 con ronda 1 | 4 | 0.7354 | 0.58375 |  0.7138 |  0.5919
Ronda 1 | 5 | 1.9179 | 0.2458 |  1.9346 | 0.257
Ronda 2 | 5 | 0.7292 | 0.5865 |  0.7316 | 0.5819
Ronda 2 con ronda 1 | 5 | 0.7292 | 0.5865 |  0.7315 |  0.5819
![Precision RN 1 softmax](http://res.cloudinary.com/chaldeacloud/image/upload/v1525478661/Proyecto%20IA/Precision_RN_1_Softmax.png)

Se puede observar que la precisión es mayor los tres casos con la función relu y además con la función softmax no varía la precisión entre la ronda 2 y la ronda 2 con ronda 1.

Los siguientes son los resultados de la segunda estructura con la función 'relu' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.8805 |0.3035 |  1.8905 | 0.2879
Ronda 2 | 1 | 0.7115 |0.6151 |  0.7344 | 0.5944
Ronda 2 con ronda 1 | 1 | 0.7109 | 0.6158 |  0.7335 |  0.5965
Ronda 1 | 2 | 1.8912 |0.2843 |  1.9274| 0.2695
Ronda 2 | 2 | 0.7024 | 0.60525 |  0.7142 | 0.5749
Ronda 2 con ronda 1 | 2 | 0.6997 |0.61 | 0.7086 | 0.5929
Ronda 1 | 3 |  1.907 |0.2775| 1.9157|0.2610
Ronda 2 | 3 | 0.7077 | 0.617 |  0.7296 | 0.5765
Ronda 2 con ronda 1 | 3 | 0.7108 |0.620| 0.7376 | 0.5725
Ronda 1 | 4|  1.8863 |0.295| 1.9323|0.2605
Ronda 2 | 4| 0.7007 | 0.6156 |  0.7073 | 0.6164
Ronda 2 con ronda 1 | 4 | 0.7076 |0.6068| 0.7186 | 0.591
Ronda 1 | 5|  1.8829 |0.2888| 1.9463|0.2650
Ronda 2 | 5| 0.7148 | 0.6087 |  0.7145 | 0.6005
Ronda 2 con ronda 1 | 5 | 0.7162 |0.6101| 0.7150 |  0.6055
![Precision RN 2 relu](http://res.cloudinary.com/chaldeacloud/image/upload/v1525486740/Proyecto%20IA/Precision_RN_2_Relu.png)

Los siguientes son los resultados de la segunda estructura con la función 'softmax' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.9255 |0.2486 |  1.9481 | 0.2580
Ronda 2 | 1 | 0.7447 |0.5798 |  0.7270 |0.584
Ronda 2 con ronda 1 | 1 | 0.7447 | 0.5798 |  0.7270 |  0.584
Ronda 1 | 2 | 1.9216 |0.25775 |  1.9027 | 0.2599
Ronda 2 | 2 | 0.7344 |0.582125 | 0.7421 |0.5785
Ronda 2 con ronda 1 | 2 | 0.7344 |0.582125 | 0.7421 |0.5785
Ronda 1 | 3 | 1.9206 |0.2528 |  1.9421 | 0.2459
Ronda 2 | 3 | 0.7416 |0.577 |  0.7234 |0.6040
Ronda 2 con ronda 1 | 3 | 0.7416 |0.577 |  0.7234 |0.6040
Ronda 1 | 4 | 1.9134 |0.2588 |  1.9144 | 0.2554
Ronda 2 | 4 | 0.7458 |0.578 | 0.7401|0.586
Ronda 2 con ronda 1 | 4 | 0.7458 |0.578 | 0.7401|0.586
Ronda 1 | 5 | 1.9232 |0.2527 |  1.9347 | 0.2665
Ronda 2 | 5 | 0.7409 |0.5753 |  0.7630 | 0.5705
Ronda 2 con ronda 1| 5 | 0.7409 |0.5753 |  0.7630 | 0.5705

![Precision RN 2 softmax](http://res.cloudinary.com/chaldeacloud/image/upload/v1525487420/Proyecto%20IA/Precision_RN_2_Softmax.png)

Al igual que con la estructura anterior, la función relu obtiene mejor precisión y con la función softmax se obtiene la misma precisión entre las predicciones de la ronda 2 y la ronda 2 con la ronda 1.
Los siguientes son los resultados de la tercer estructura con la función 'relu' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.8774 |0.2825 |  1.9201 | 0.2724
Ronda 2 | 1 | 0.6974 |0.6251| 0.7216|0.5975
Ronda 2 con ronda 1 | 1 | 0.6960 |0.6223 |  0.7228 |  0.5849
Ronda 1 | 2 | 1.8900 |0.2833 |  1.9019 | 0.2709
Ronda 2 | 2 | 0.7157 |0.6132| 0.7332|0.5930
Ronda 2 con ronda 1 | 2 | 0.7131 |0.6175 |  0.7324|  0.5905
Ronda 1 | 3 | 1.9006 |0.2851 |  1.8944 | 0.2770
Ronda 2 | 3 | 0.7064 |0.6135|  0.7172|0.6145
Ronda 2 con ronda 1 | 3 | 0.7056 |0.6118 |  0.7152| 0.6045
Ronda 1 | 4 | 1.8832 |0.283|  1.8923 |0.2740
Ronda 2 | 4 | 0.7033 |0.6245|  0.7200|0.6120
Ronda 2 con ronda 1 | 4 | 0.6997 |0.6303|   0.7171|  0.616
Ronda 1 | 5 | 1.8911 | 0.2842|  1.9263 |0.2834
Ronda 2 | 5 | 0.7033 |0.6103|  0.7317|0.5935
Ronda 2 con ronda 1 | 5 | 0.6995 |0.6121|  0.7312|  0.5975

![Precision RN 3 Relu](http://res.cloudinary.com/chaldeacloud/image/upload/v1525488403/Proyecto%20IA/Precision_RN_3_Relu.png)

Los siguientes son los resultados de la tercer estructura con la función 'softmax' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 1.9308 |0.26225| 1.9584 | 0.2455
Ronda 2 | 1 | 0.7331 |0.5818| 0.7320|0.5925
Ronda 2 con ronda 1 | 1 | 0.7327 |0.5818 |  0.7315 | 0.5925
Ronda 1 | 2 | 1.9331 |0.2583| 1.9127 |0.2659
Ronda 2 | 2 | 0.7343 |0.5807| 0.7213|0.5880
Ronda 2 con ronda 1 | 2 | 0.7343 |0.5807| 0.7213|0.5880
Ronda 1 | 3 | 1.9271 |0.2573| 1.9361 |0.2415
Ronda 2 | 3 | 0.7426 |0.5881| 0.7547|0.5815
Ronda 2 con ronda 1| 3 | 0.7426 |0.5881| 0.7547|0.5815
Ronda 1 | 4 | 1.9172 |0.2572|1.9133 |0.2665
Ronda 2 | 4 | 0.7363 |0.5718|0.7400|0.5875
Ronda 2 con ronda 1 | 4 | 0.7363 |0.5718|0.7400|0.5875
Ronda 1 | 5 | 1.9162 |0.2546|1.9186 |0.257
Ronda 2 | 5 | 0.7393 |0.5845|0.7237|0.5980
Ronda 2 con ronda 1| 5 | 0.7393 |0.5845|0.7237|0.5980

![Precision RN 3 softmax](http://res.cloudinary.com/chaldeacloud/image/upload/v1525489344/Proyecto%20IA/Precision_RN_3_softmax.png)

## SVM

Para el uso de SVM(Support Vector Machine) se parametrizó el uso de kernels mediante la bandera --p-kernel la cual puede ser cualquiera de los valores aceptados por la clase "svm" de sklearn.
En este análisis se probaron dos kernels distintos:
* 'linear'
* 'rbf'

Los siguientes son los resultados del kernel 'linear' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:

Predicción  |Iteración| Error en entrenamiento| Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 41.6951 | 0.2851| 42.344 | 0.2675
Ronda 2 | 1 | 0.4548|0.5943| 0.4735|0.5735
Ronda 2 con ronda 1 | 1 | 0.4548 |0.5943 |  0.4735 | 0.5735
Ronda 1 | 2 | 39.63 | 0.2755| 40.35 | 0.267
Ronda 2 | 2 | 0.4262|0.60975| 0.4005|0.6255
Ronda 2 con ronda 1 | 2 | 0.42625 |0.60975 |  0.4025 | 0.6235
Ronda 1 | 3 | 38.0358 | 0.2865| 38.4955 | 0.2755
Ronda 2 | 3 | 0.4442|0.6035| 0.4415| 0.5985
Ronda 2 con ronda 1 | 3 | 0.4442 |0.6035 |  0.4415 | 0.5985
Ronda 1 | 4 | 42.303 | 0.274| 39.646 | 0.2815
Ronda 2 | 4 |  0.42775| 0.6066 |0.405| 0.608
Ronda 2 con ronda 1 | 4 |  0.4277 |0.6066 |  0.405| 0.608
Ronda 1 | 5 | 42.2571 |0.2796| 43.262 | 0.2875
Ronda 2 | 5 | 0.4300|  0.605 |0.4335|  0.602
Ronda 2 con ronda 1 | 5 |  0.4306 |0.6050 | 0.4335| 0.602

![Precision SVM Linear](http://res.cloudinary.com/chaldeacloud/image/upload/v1525490588/Proyecto%20IA/Precision_SVM_Linear.png)

Los siguientes son los resultados del kernel 'rbf' con 10000 muestras y 80% del conjunto de datos utilizado como entrenamiento:
Predicción  |Iteración| Error en entrenamiento| Precision en entrenamiento | Error en pruebas | Precision en pruebas
-------------|----------------------|----------------------|----------------------------|------------------|--------------------|
Ronda 1 | 1 | 33.6551 | 0.298 | 35.729 | 0.2745
Ronda 2 | 1 | 0.4401|0.6028|  0.4400|0.5930
Ronda 2 con ronda 1 | 1 | 0.4376 |0.6046 |  0.4420 |  0.5925
Ronda 1 | 2 | 38.5168 | 0.2913 | 37.637 | 0.2745
Ronda 2 | 2 | 0.4335|0.6025|  0.4425| 0.5960
Ronda 2 con ronda 1 | 2 | 0.4338 |0.6028| 0.4410 |  0.5975
Ronda 1 | 3 | 34.2633 |0.2917 | 35.0515 | 0.2890
Ronda 2 | 3 | 0.4380 | 0.6052|  0.4165| 0.6220
Ronda 2 con ronda 1 | 3 | 0.4363 |0.6068| 0.4200 |  0.6185
Ronda 1 | 4 | 39.1548 |0.2870 | 36.7915 | 0.2800
Ronda 2 | 4 | 0.4248|0.6191| 0.4230 | 0.6085
Ronda 2 con ronda 1 | 4 | 0.4241 |0.6202| 0.4245|  0.6070
Ronda 1 | 5 | 34.3276 |0.2961 | 35.9885| 0.2840
Ronda 2 | 5 | 0.421625|0.6132| 0.4215 | 0.6155
Ronda 2 con ronda 1 | 5 | 0.4221 |0.6131|0.4220|  0.6150

![Precision SVM rbf](http://res.cloudinary.com/chaldeacloud/image/upload/v1525491538/Proyecto%20IA/Precision_SVM_rbf.png)

## Árboles de Decisión
Para la prediccion de los votos de los votantes generados en el simulador descrito anteriormente, se implementó un método de clasificación, utilizando árboles de decisión mediante una implementación propia, es decir, sin el uso de alguna librería que permita la fácil contrucción del árbol, el cuál dados los indicadores de los votantes predice en el caso de la primera ronda por cuál de los 13 partidos votó el votante generado o bien si su voto fué nulo o blanco y en el caso de la segunda ronda por cuál de los dos partidos fué su voto o bien si el mismo fué nulo o blanco.
Como parte del análisis de resultados del método del árbol de decisión, es importante analizar los siguientes aspectos:
* Se tomó como referencia el algoritmo que se encuentra en el libro de Stuart Russell y Peter Norvig: Artificial Intelligence, A Modern Approach, sin embargo, se mejoró la manera de crear los nodos hijos, es decir, en lugar de tomar en cuenta todas las ganancias desde el principio del algoritmo, se calculan las ganancias cada vez que se haga la partición de un camino para crear nuevos nodos hijos, así, las nuevas ganancias caluladas se centran en los nuevos conjuntos que se obtienen al particionar el conjunto del padre.
* Las ganancias son calculadas de la manera en que son explicadas en el libro mencionado en el punto anterior, esto gracias a varias funciones: obtener_entropia_conjunto_entrenamiento, la cual recibe el conjunto de entrenamiento de entrada y obtener_entropia_conjunto_entrenamiento_formula, la cual se encarga de utilizar la fórmula de la entropía. Se implementó una función llamada recorrer_columnas_datos_entrenamiento, encargada de obtener las ganancias por columna del conjunto de entrenamiento, esta tiene conexión con una función llamada obtener_ganancia_columna, la cual obtiene la ganancia para una columna específica, esta a su vez, utiliza la función resultado_logaritmico_probabilidad para aplicar la fórmula de la ganancia para la columna.
* Para el armado del árbol, se utiliza la recursividad, hasta que se cumpla uno de los tres casos de parada del algoritmo mencionado en el libro, o bien, se llama de nuevo a la función de armado del árbol si se debe hacer una partición para crear nuevos nodos hijos.
* Se hizo uso de variables globales que puedan contener los encabezados del conjunto de entrenamiento, además de controlar cuáles atributos ya habían sido utilizados para crear nuevos nodos.
* Se implementaron pequeñas funciones, de manera que puedan funcionar como módulos que faciliten los cálculos necesarios para armar el árbol, algunas de estas funciones son: reducir_ganancias_por_atributos_utilizados, la cual reduce la cantidad de atributos que se puedan utilizar para crear un nuevo nodo, tomando en cuenta los atributos que ya fueron utilizados para el mismo fin, obtener_filas_mayor_menor_cero, la cual crea una partición para datos de entrada normalizados, de manera que se puedan separar los que son mayores a cero y menores a cero, seguidamente, se pueden crear dos nuevos nodos hijos que sirvan como nuevos caminos, son_todos_target_igualles, la cual se encarga de determinar si todos los atributos actuales se relacionan con la misma etiqueta, de  manera que se pueda validar uno de los casos de parada de la función recursiva de armado del árbol.
* La función encargada de armar el árbol de decisión se llama armar_arbol, esta se encarga de seguir los pasos dados en el libro, con la ligera variación que se menciona en el primer punto.
* Existe una función que es llamada previa a la de armar_arbol, esta se llama generar_arbol, se encarga de hacer la partición de datos de entrenamiento y de pruebas, de manera que los datos de entrenamiento sean utilizados para crear un nuevo árbol de decisión.
* Para hacer una predicción, se implementó una función llamada predecir, la cual recibe como parámetro el conjunto de datos que se quiere predecir y el árbol de decisión con el que se hará dicha acción. Esta a su vez, hace un llamado a una función auxiliar, la cual se encarga de recorrer el árbol, basándose en los datos de cada fila de los datos que quieren ser predecidos, hasta llegar a un nodo hoja, el cual contendrá la predicción para esa fila en específico.
* Después de una predicción, se hace el cálculo de los verdaderos positivos y falsos positivos, para que se puedan dar datos referentes a la precisión, errores de prueba y errores de entrenamiento. Para esto, se implementaron funciones como: obtener_verdaderos_falsos_positivos, la cual recibe los valores para los cuales se hizo predicción y los valores reales, de manera que, comparando estos dos parámetros, se puedan clasificar en verdaderos positivos y falsos positivos. Para el cálculo de precisión se utiliza una función llamada obtener_precision, la cual toma como parámetro los verdaderos positivos y falsos positivos, seguidamente, utilizará la fórmula de precisión para retornar el resultado al aplicarla.
* Para los errores de entrenamiento y de pruebas, se utilizó la validación 0/1, es decir sumar un punto al error cada vez que un valor que se haya predecido no coincida con el valor real, es decir, la suma de los falsos positivos. 
* Debida a la ligera modificación en la implementación del árbol de decisión, el resultado, sin poda, es un árbol casi óptimo, el cual aún apllicándole la poda no mejorará su precisión significativamente, sin embargo, si hace un pequeño porcentaje de  mejora en algunas pruebas, en otras pruebas se mantiene la precisión o baja muy poco. El análisis del árbol de decisión para diferentes umbrales se explicará en breve.
* La poda se hizo de acuerdo a un umbral que se recibe como parámetro, por lo que, el árbol de decisión se estructuró con dos subestructuras: nodo y hoja. La subestructura nodo se encarga de guardar la ganancia para un atributo, así como sus hijos y las filas relacionadas a la columna, además del número de columna con la que se representa el nodo. Por otro lado, la subestructura hoja, se encarga de almacenar la etiqueta que está prediciendo, así, al predecir un conjunto de datos, en el momento en el que se llegue a una hoja, se retorna la etiqueta como la predicción. Sabiendo esta estructura, una vez que se recibe un umbral como parámetro, se buscarán los nodos que solo tengan hojas como hijos, se evaluará su ganancia, y se concluirá si es candidato o no a la poda. Para esto, se utiliza la función obtener_nodos_con_solo_hojas, de manera que puedan ser evaluados para podar el árbol.

Utilizando un 80% del conjunto de datos como muestras de entrenamiento, se obtuvieron los siguientes resultados:

Predicción |Tamaño de muestra| Error en entrenamiento | Precision en entrenamiento | Error en pruebas | Precision en pruebas
-----------|---------------|------------------------|----------------------------|------------------|--------------------|
Ronda 1 | 10000 | Sin umbral | 5992 | 0.251 | 1457 | 0.2175
Ronda 2 | 10000 | Sin umbral | 3314 | 0.58575 |  821 | 0.5895
Ronda 2 con Ronda 1 | 10000 | Sin umbral | 3313 | 0.585875 |  821 | 0.585875
Ronda 1 | 10000 | 0.08 | 5893 | 0.263375 | 1474 | 0.263
Ronda 2 | 10000 | 0.08 | 3332 | 0.5835 |  870 | 0.565
Ronda 2 con Ronda 1 | 10000 | 0.08 | 3332 | 0.5835 |  870 | 0.565
Ronda 1 | 10000 | 0.06 | 5929 | 0.258875 | 1480 | 0.26
Ronda 2 | 10000 | 0.06 | 3325 | 0.584375 |  827 | 0.5865
Ronda 2 con Ronda 1 | 10000 | 0.06 | 3323 | 0.584625|  826 | 0.587
Ronda 1 | 10000 | 0.04 | 5917 | 0.260375 | 1442 | 0.279
Ronda 2 | 10000 | 0.04 | 3345 | 0.581875 |  871 | 0.5645
Ronda 2 con Ronda 1 | 10000 | 0.04 | 3346 | 0.58175|  872 | 0.564
Ronda 1 | 10000 | 0.1 | 5977 | 0.252875 | 1467 | 0.2665
Ronda 2 | 10000 | 0.1 | 3252 | 0.5935 |  840 | 0.58
Ronda 2 con Ronda 1 | 10000 | 0.1 | 3252 | 0.5935|  840 | 0.58
Ronda 1 | 10000 | 0.15 | 5970 | 0.25375 | 1474 | 0.263
Ronda 2 | 10000 | 0.15 | 3325 | 0.584375 |  842 | 0.579
Ronda 2 con Ronda 1 | 10000 | 0.15 | 3325 | 0.584375|  842 | 0.579


Los datos presentados anteriormente se pueden visualizar de forma gráfica en los siguientes gráficos:

![Árbol de decisión sin umbral](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/sin_umbral.png)
![Árbol de decisión con umbral = 0.08](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/con_008.png)
![Árbol de decisión con umbral = 0.06](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/con_006.png)
![Árbol de decisión con umbral = 0.04](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/con_004.png)
![Árbol de decisión con umbral = 0.1](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/con_1.png)
![Árbol de decisión con umbral = 0.15](https://res.cloudinary.com/poppycloud/image/upload/v1525489797/con_15.png)

Dados los resultados anteriores se puede concluir lo siguiente:
* La predicción de la primera ronda tiende a dar valores de precisión bajos debido a la cantidad de posibles salidas (15 opciones por voto).
* La implementación de un umbral logra que la precisión suba un poco, se mantenga o baje un poco, debido a que el primer árbol generado (sin poda) ya posee una precisión óptima para este tipo de modelo.
* Conforme se aumenta el umbral, los datos de la segunda ronda se parecen más a los de la segunda ronda más primera ronda.
* En el mejor de los casos, el modelo tiene una precisión un poco menor a 60%, lo cual se debe a varios aspectos:
  *La naturaleza de los datos (son simulados).
  *Algunos datos los cuales se creen ayudarían a la generalización no fueron muestreados ya que eran promedios, por ejemplo: el promedio de escolaridad.
  *No se conoce la distribución verdadera de cada columna, por lo cual se opta por normalizar con media de 0 y desviación 1 todos los datos.

## KNN con kd-tree
### Descripción general de algoritmo base

El algoritmo se basa en la creación de un árbol binario equilibrado sobre datos con un número arbitrario de dimensiones. Para la creación del árbol se siguen los siguientes pasos:

* Se escoge una dimensión.
* Se escoge una media del set de datos basado en la dimensión. 
* Se dividen los datos basado en la media.
* Se envía la mitad de los datos a la rama izquierda o derecha según la media.
* Se repite el proceso en cada rama hasta tener en cada nodo 2 o menos ejemplos de datos.

Para buscar los vecinos más cercanos con kd-tree se sigue el siguiente proceso:

* Se toma la dimensión del nodo que se está evaluando.
* Se toma el valor del dato que se quiere clasificar basado en la dimensión.
* Si el valor del dato es más mayor o igual que el valor del nodo, se recorre el árbol por la derecha de lo contrario se va a la izquierda.
* Se repite el proceso hasta llegar a las hojas, y por pluralidad se decide a que categoría pertenece el dato que se desea clasificar.


### Implementación
A lo largo del desarrollo del algoritmo se abarcaron diferentes enfoques, con resultados variados, en esta sección se explicará cada uno de esos enfoques.

##### Consideraciones previas

Primero es necesario detallar ciertas consideraciones que se tuvieron antes de explicar los enfoques.

Considera la siguiente imagen, si se toma en cuenta que, en el algoritmo explicado anteriormente la condición de parada de creación del árbol es que haya 2 o menos ejemplos (círculos verdes) como se puede observar en la figura 1. Ahora considere que el algoritmo visualiza 4 vecinos cercanos para hacer una categorización, si tomamos en cuenta la figura 1 como árbol, el algoritmo tendría que tomar ambos caminos del nodo A e ir por el nodo B y C para obtener los 4 vecinos como se visualiza en la figura 2.  Una forma mas sencilla de resolver lo anterior es alterar la condición de parada del algoritmo base y parametrizarlo con la cantidad de vecinos que se desean considerar, esto ahorraría la creación niveles innecesarios en el árbol, y simplifica la búsqueda de vecinos, esto se puede visualizar en la figura 3.

![KD_Tree](https://i.imgur.com/8BYdSY0.png)

Todos los enfoques siguen la estructura del algoritmo base lo que se cambio fue la forma de escoger, dimensiones, de dividir los datos, de sacar la media, de escoger vecinos más cercanos, etc. Además, en este sección solo se muestra resultados de predicción de r2 con r1.

#### Enfoque 1
A continuación, se explica cómo se desarrolló cada elemento que toma en cuenta el algoritmo de kd-tree:

* Escoger dimensión: Se implemento con una función random
* Media: Se sumaron los valores de todos datos ejemplos según la dimensión obtenida, y luego se dividió por la cantidad de datos.
* Dividir datos: Si los datos eran menores a la media se ponían en rama izquierda sino iban a la derecha.
* Escoger KNN: Se recorría el árbol siguiendo el algoritmo base.
* Creación de nodos del árbol: Los datos en el nodo solo tomaba en cuenta el valor correspondiente a la dimensión, por ejemplo: 
    * Data = [1,2,3]
    * Dimensión = 0
    * Data en nodo= 1

##### Ejemplo de una corrida

Cantidad de datos=15000, k=150, 40% de los datos era para validación, se estaba prediciendo segunda ronda con primera:

El valor de error fue aproximadamente 0.94

##### Observaciones

* Mucha vez el random escogía la misma dimensión y no era objetivo a la hora de dividir los datos.
* Se noto que k debía ser la cantidad de datos entre 100, sino el programa se caía por exceder la cantidad de recursiones. Luego se observo que esto se debía a la forma en la que se seleccionaba la media y se dividían los valores, ya que ocasionaba que muchas ramas quedaran vacías y la mayoría de los datos se fueran a la rama derecha haciendo que los niveles del árbol creciesen exponencialmente.
* Muchas veces la media tendía a ser 0 porque al normalizar los datos quedaba muchos valores con 0.
* La división de los datos no era equilibrada y terminaba con 60%-90% de los datos en una de las ramas.
* En este enfoque se decidió como prueba ignorar las medias que fueran 0 y el valor de error bajo considerablemente dando como resultado 0.57 tomando en cuenta que k=150, n=15000 y porcentaje=40%.

#### Enfoque 2

A continuación, se explica cómo se desarrolló cada elemento que toma en cuenta el algoritmo de kd-tree:

* Escoger dimensión: Se implemento con la siguiente operación “i mod n”, donde i es el nivel del árbol y n la cantidad de dimensiones.
*	Media: Se ordenaron los datos de menor a mayor, y se tomo el dato que se encontraba en la posición central.
*	Dividir datos: Tomando los datos ordenados de menor a mayor, los que se encontraban antes de la media se mandaba a la rama izquierda y los demás a la derecha.
*	Escoger KNN: Se recorría el árbol siguiendo el algoritmo base.
*	Creación de nodos del árbol: Los datos en el nodo solo tomaba en cuenta el valor correspondiente a la dimensión, por ejemplo: 
    *	Data = [1,2,3]
    *	Dimensión = 0
    *	Data en nodo= 1

##### Ejemplo de una corrida

Cantidad de datos=15000, k=150, 40% de los datos era para validación, se estaba prediciendo segunda ronda con primera:

El valor de error fue aproximadamente 0.42

##### Observaciones

*	El problema de este enfoque es que no era consistente, ya que a veces el valor del error daba 1. El valor de error estaba muy relacionado al tamaño de k y cuantos datos se usaba para entrenamiento. Si se usaba un k pequeño el porcentaje de datos de validación debía ser bajo y si k era grande el porcentaje debía ser mayor. Por ejemplo:
    *	n=15000, k=1500, porcentaje=20. Error = 1
    *	n=15000, k=150, porcentaje=20. Error = 0.4

#### Enfoque 3

A continuación, se explica cómo se desarrolló cada elemento que toma en cuenta el algoritmo de kd-tree:
*	Escoger dimensión: Se implemento con la siguiente operación “i mod n”, donde i es el nivel del árbol y n la cantidad de dimensiones.
*	Media: Se ordenaron los datos de menor a mayor, y se tomó el dato que se encontraba en la posición central.
*	Dividir datos: Tomando los datos ordenados de menor a mayor, los que se encontraban antes de la media se mandaba a la rama izquierda y los demás a la derecha.
*	Escoger KNN: Se recorría el árbol siguiendo el algoritmo base, pero además se hace una comparación con la rama opuesta. La comparación consiste en medir la distancia entre las ramas y el punto a categorizar, y devolvía la rama con menor distancia
*	Creación de nodos del árbol: Los datos en el nodo solo tomaba en cuenta el valor correspondiente a la dimensión, por ejemplo: 
    *	Data = [1,2,3]
    *	Dimensión = 0
    *	Data en nodo= [1,2,3]
    
##### Ejemplo de una corrida

Cantidad de datos=15000, k=150, 40% de los datos era para validación, se estaba prediciendo segunda ronda con primera:

El valor de error fue aproximadamente 0.41

##### Observaciones
*	Los resultados de error del algoritmo no varían demasiado, el tamaño de k y cantidad de datos no afecta en gran medida la clasificación de datos y los resultados se mantiene estables. El valor de error da entre 0.71-0.75 para r1 y entre 0.41-0.49 para r2 y r1_con_r2.

#### Conclusión

El enfoque 3 fue el que dio mejores resultados porque a pesar de que el valor de error era muy similar al enfoque 2, el enfoque 3 fue mucho mas constante en sus resultados. 

En el enfoque 1 al ignorar las medias que fueran 0, mejoro mucho los resultados del algoritmo sin embargo era susceptible a k pequeñas lo cual provocaba que el programa fuera inestable. El principal problema de este algoritmo fue la división de datos que generaban que la profundidad del árbol incrementara. 

Con el enfoque 2 el principal problema era la constancia de resultados porque podía dar 1 de error o un valor más bajo como 0.41

En general el tiempo de ejecución de los 3 enfoques fue muy similar. Utilizando el enfoque 3, se tardó 7 segundos con un dataset de tamaño 10000, k=10, y porcentaje=20. La prueba mas grande que se realizo fue con un dataset de tamaño 100000, k=100, y porcentaje=20, esta prueba tardo 1 minuto con 20 segundos. A continuación, se muestran los resultados de ambas ejecuciones.

![Run1](https://i.imgur.com/eL8RZB7.png)
![Run2](https://i.imgur.com/HOzJ49r.jpg)

# Instalación de las herramientas
Si bien, los módulos tec.ic.ia.pc1.g05 y tec.ic.ac.p1.g05 al ser importados, se instalarán todas las dependencias del proyecto, se listarán los comandos necesarios (pip) para instalar las herramientas manualmente.
* pip install numpy
* pip install sklearn
* pip install keras
* pip install tensorflow
* pip install pandas
* pip install scipy

