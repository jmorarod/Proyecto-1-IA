from tec.ic.ia.pc1.g05 import generar_muestra_pais, generar_muestra_provincia
#from pc1 import generar_muestra_pais, generar_muestra_provincia
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import keras
from keras.optimizers import SGD
import tensorflow as tf
import pandas as pd

def split_muestra(muestra, porcentaje):
    cantidad_test = int(len(muestra) * (porcentaje / 100))
    cantidad_training = len(muestra) - cantidad_test
    test_set = []
    training_set = []
    for i in range(0, len(muestra)):
        if(i < cantidad_training):
            training_set += [muestra[i]]
        else:
            test_set += [muestra[i]]
    return training_set, test_set

def regresion_logistica_r1(train_set, test_set, learning_rate, regularizacion):
    num_epochs = 1500
    display_step = 1
    x_train = []
    x_test = []
    y_test = get_column(test_set,-1)
    y_train = get_column(train_set,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    for i in range(0,len(y_test)):
        y_test[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 15)
    y_test = keras.utils.to_categorical(y_test,num_classes = 15)
    for i in range (0, len(train_set)):
        x_train += [train_set[i][:-1]]
    for i in range (0, len(test_set)):
        x_test += [test_set[i][:-1]]

    sess = tf.InteractiveSession()
   
    x = tf.placeholder("float",[None, 22])
    y = tf.placeholder("float",[None, 15])
    W = tf.Variable(tf.zeros([22,15]))
    b = tf.Variable(tf.zeros([15]))

    sess.run(tf.initialize_all_variables())
    y_ = tf.nn.softmax(tf.matmul(x,W)+b)

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    if(regularizacion == "l1"):
                        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                                                      l1_regularization_strength=0.5).minimize(cost)
    else:
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                                                      l2_regularization_strength=0.5).minimize(cost)
                        
    
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        cost_in_each_epoch = 0
        for epoch in range(num_epochs):
           
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
            cost_in_each_epoch += c
        cost_in_each_epoch_2 = 0
        for epoch in range(num_epochs):

            _, c2 = sess.run([optimizer, cost], feed_dict={x: x_test, y: y_test})
            cost_in_each_epoch_2 += c2
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        cost_train = 0
        cost_test = 0
        for i in cost_in_each_epoch:
            cost_train += i
        for i in cost_in_each_epoch_2:
            cost_test += i
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Error en entrenamiento: ", cost_train)
        print("Precision en entrenamiento: ", accuracy.eval({x: x_train, y: y_train}))
        print("Error en pruebas: ", cost_test)
        print("Precision en pruebas: ", accuracy.eval({x: x_test, y: y_test}))
        sess.close()
    
def regresion_logistica(train_set, test_set, learning_rate, regularizacion, r2_con_r1=False):
    num_epochs = 1500
    display_step = 1
    x_train = []
    x_test = []
    y_test = get_column(test_set,-1)
    y_train = get_column(train_set,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    for i in range(0,len(y_test)):
        y_test[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 4)
    y_test = keras.utils.to_categorical(y_test,num_classes = 4)
    for i in range (0, len(train_set)):
        x_train += [train_set[i][:-1]]
    for i in range (0, len(test_set)):
        x_test += [test_set[i][:-1]]

    sess = tf.InteractiveSession()
    if(r2_con_r1):
        x = tf.placeholder("float",[None, 23])
    else:
        x = tf.placeholder("float",[None, 22])
    y = tf.placeholder("float",[None, 4])
    if(r2_con_r1):
        W = tf.Variable(tf.zeros([23,4]))
    else:
        W = tf.Variable(tf.zeros([22,4]))
    b = tf.Variable(tf.zeros([4]))

    sess.run(tf.initialize_all_variables())
    y_ = tf.nn.softmax(tf.matmul(x,W)+b)

    cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    if(regularizacion == "l1"):
                        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                                                      l1_regularization_strength=1.0).minimize(cost)
    else:
        optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate,
                                                                      l1_regularization_strength=1.0).minimize(cost)
    with tf.Session() as sess:
        
        sess.run(tf.global_variables_initializer())
        for epoch in range(num_epochs):
            cost_in_each_epoch = 0
           
            _, c = sess.run([optimizer, cost], feed_dict={x: x_train, y: y_train})
            cost_in_each_epoch += c
        cost_in_each_epoch_2 = 0
        for epoch in range(num_epochs):

            _, c2 = sess.run([optimizer, cost], feed_dict={x: x_test, y: y_test})
            cost_in_each_epoch_2 += c2
        correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
        cost_train = 0
        cost_test = 0
        for i in cost_in_each_epoch:
            cost_train += i
        for i in cost_in_each_epoch_2:
            cost_test += i
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Error en entrenamiento: ", cost_train)
        print("Precision en entrenamiento: ", accuracy.eval({x: x_train, y: y_train}))
        print("Error en pruebas: ", cost_test)
        print("Precision en pruebas: ", accuracy.eval({x: x_test, y: y_test}))
        sess.close()


def redes_neuronales(n_muestra, porcentaje_test, num_capas, unidades_por_capa, activacion):
    muestra = generar_muestra_pais(n_muestra)
    muestra_r1 = datos_r1_normalizados(muestra)
    muestra_r2 = datos_r2_normalizados(muestra)
    muestra_r2_r1 = datos_r2_con_r1_normalizados(muestra)
    train_r1, test_r1 = split_muestra(muestra_r1, porcentaje_test)
    train_r2, test_r2 = split_muestra(muestra_r2, porcentaje_test)
    train_r2_r1, test_r2_r1 = split_muestra(muestra_r2_r1, porcentaje_test)
    predicciones_train_r1, predicciones_test_r1 = red_neuronal_r1(train_r1, test_r1, num_capas, unidades_por_capa, activacion)
    predicciones_train_r2, predicciones_test_r2 = red_neuronal_r2(train_r2, test_r2, num_capas, unidades_por_capa, activacion)
    predicciones_train_r2_r1, predicciones_test_r2_r1 = red_neuronal_r1_r2(train_r2_r1, test_r2_r1, num_capas, unidades_por_capa, activacion)
    for i in range(0, len(train_r1)):
        muestra[i] += [True, predicciones_train_r1[i],predicciones_train_r2[i],predicciones_train_r2_r1[i]]
    for i in range(0, len(test_r1)):
        muestra[i+len(train_r1)] += [False, predicciones_test_r1[i]+1,predicciones_test_r2[i]+1,predicciones_test_r2_r1[i]+1]
    dataframe = pd.DataFrame(muestra,columns=['poblacion_canton', 'superficie_canton','densidad_poblacion','urbano','sexo','dependencia_demografica','ocupa_vivienda','promedio_ocupantes','vivienda_buen_estado', 'vivienda_hacinada','alfabetismo','escolaridad_promedio','educacion_regular','fuera_fuerza_trabajo','participacion_fuerza_trabajo','asegurado','extranjero','discapacidad','no_asegurado', 'porcentaje_jefatura_femenina','porcentaje_jefatura_compartida', 'edad','voto_primera_ronda','voto_segunda_ronda','es_entrenamiento', 'prediccion_r1', 'prediccion_r2','prediccion_r2_con_r1'])
    dataframe.to_csv('resultados_redes_neuronales.csv',index=False)
    return muestra
        

def red_neuronal_r1_r2(muestra, test, num_capas, unidades_por_capa, activacion):
    x_train = []
    x_test = []
    y_test = get_column(test,-1)
    y_train = get_column(muestra,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    for i in range(0,len(y_test)):
        y_test[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 4)
    y_test = keras.utils.to_categorical(y_test,num_classes = 4)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    for i in range (0, len(test)):
        x_test += [test[i][:-1]]    
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=23))
    for i in range(1,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(4,activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128,
               validation_data =(x_test,y_test),
               shuffle = True,
               verbose=0)
    predicciones_train = modelo.predict(x_train,
                                        batch_size=128,
                                        verbose=0)
    predicciones_test =  modelo.predict(x_test,
                                        batch_size=128,
                                        verbose=0)
    loss_acc_train = modelo.evaluate(x_train, y_train,
                                     batch_size=128,
                                     verbose=0)
    print("[============================]")
    print("Ronda 2 con Ronda 1 - Error en training: ", loss_acc_train[0])
    print("Ronda 2 con Ronda 1 - Precision en training: ", loss_acc_train[1])
    loss_acc_test = modelo.evaluate(x_test, y_test,
                                    batch_size=128,
                                    verbose=0)
    print("[============================]")
    print("Ronda 2 con Ronda 1 - Error en test: ", loss_acc_test[0])
    print("Ronda 2 con Ronda 1 - Precision en test: ", loss_acc_test[1])
    predicciones_train = convert_from_one_hot(predicciones_train)
    predicciones_test = convert_from_one_hot(predicciones_test)
    return predicciones_train, predicciones_test

def red_neuronal_r2(muestra, test, num_capas, unidades_por_capa, activacion):
    
    x_train = []
    y_train = get_column(muestra,-1)
    x_test = []
    y_test = get_column(test,-1)
    
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    for i in range(0,len(y_test)):
        y_test[i] -= 1
    y_test = keras.utils.to_categorical(y_test,num_classes = 4)
    y_train = keras.utils.to_categorical(y_train,num_classes = 4)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    for i in range (0, len(test)):
        x_test += [test[i][:-1]] 
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=22))
    for i in range(1,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(4,activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128,
               validation_data = (x_test, y_test),
               verbose=0,
               shuffle = True
               )
    predicciones_train = modelo.predict(x_train, batch_size=128)
    predicciones_test =  modelo.predict(x_test, batch_size=128)
    loss_acc_train = modelo.evaluate(x_train, y_train,
                                     batch_size=128,
                                     verbose=0)
    print("[============================]")
    print("Ronda 2 - Error en training: ", loss_acc_train[0])
    print("Ronda 2 - Precision en training: ", loss_acc_train[1])
    loss_acc_test = modelo.evaluate(x_test, y_test,
                                    batch_size=128,
                                    verbose=0)
    print("[============================]")
    print("Ronda 2 - Error en test: ", loss_acc_test[0])
    print("Ronda 2 - Precision en test: ", loss_acc_test[1])
    predicciones_train = convert_from_one_hot(predicciones_train)
    predicciones_test = convert_from_one_hot(predicciones_test)
    return predicciones_train, predicciones_test

#La longitud de la lista unidades_por_capa tiene que ser num_capas + 1
def red_neuronal_r1(muestra, test, num_capas, unidades_por_capa, activacion): 
    x_train = []
    y_train = get_column(muestra,-1)
    x_test = []
    y_test = get_column(test,-1)
    for i in range(0,len(y_train)):
        y_train[i] -= 1
    for i in range(0,len(y_test)):
        y_test[i] -= 1
    y_train = keras.utils.to_categorical(y_train,num_classes = 15)
    y_test = keras.utils.to_categorical(y_test,num_classes = 15)
    for i in range (0, len(muestra)):
        x_train += [muestra[i][:-1]]
    for i in range (0, len(test)):
        x_test += [test[i][:-1]] 
    modelo = Sequential()
    modelo.add(Dense(units=unidades_por_capa[0],activation='relu',input_dim=22))
    for i in range(0,num_capas):
        modelo.add(Dense(units=unidades_por_capa[i],activation=activacion))
    modelo.add(Dense(15,activation='softmax'))
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    modelo.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    modelo.fit(x_train, y_train,
               epochs = 20,
               batch_size = 128,
               validation_data =(x_test,y_test),
               shuffle = True,
               verbose=0 )
    predicciones_train = modelo.predict(x_train, batch_size=128)
    predicciones_test =  modelo.predict(x_test, batch_size=128)
    loss_acc_train = modelo.evaluate(x_train, y_train,
                                     batch_size=128,
                                     verbose=0)
    print("[============================]")
    print("Ronda 1 - Error en training : ", loss_acc_train[0])
    print("Ronda 1 - Precision en training : ", loss_acc_train[1])
    loss_acc_test = modelo.evaluate(x_test, y_test,
                                    batch_size=128,
                                    verbose=0)
    print("[============================]")
    print("Ronda 1 - Error en test : ", loss_acc_test[0])
    print("Ronda 1 - Precision en test : ", loss_acc_test[1])

    predicciones_train = convert_from_one_hot(predicciones_train)
    predicciones_test = convert_from_one_hot(predicciones_test)
    return predicciones_train, predicciones_test
        
        

def convert_from_one_hot(one_hot_output):
    num_classes = len(one_hot_output[0])
    outputs = []
    for output in one_hot_output:
        max_output = 0
        max_output_index = 0
        for j in range(0, num_classes):
            if(output[j] >= max_output):
                max_output = output[j]
                max_output_index = j
        outputs += [max_output_index]
    return outputs
        
    
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
