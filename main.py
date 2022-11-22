#-----------------------------------------------------#
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
print(tf.__version__)
import sqlite3
#-----------------------------------------------------#
try:
    mi_conexion=sqlite3.connect("database.db")
    cursor=mi_conexion.cursor()
    cursor.execute("SELECT PH,FECHAREGISTRO FROM TBL_DATOS_SENSOR")
    rows=cursor.fetchall()
    
    ph=[]
    lista=[]
    for row in rows:
        ph.append(row[0])
        lista.append(row[1][6:10].replace('-','.'))
    
    ph_final=np.array(ph,dtype=int)
    lista_final=np.array(lista,dtype=float)
    
    capa1=tf.keras.layers.Dense(units=3,input_shape=[1])
    capa2=tf.keras.layers.Dense(units=3)
    salida=tf.keras.layers.Dense(units=1)
    modelo=tf.keras.Sequential([capa1,capa2,salida])
    
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='mean_squared_error',metrics=['accuracy']
    )
    historial=modelo.fit(lista_final,ph_final, epochs=30, verbose=False)
    
    resultado= round(float(modelo.predict([8.20])), 3) ## MES Y DIA
    
    print(resultado)
    
    #-----------------------------------------------------#    
    #GRAFICO DE ENTRENAMIENTO
    plt.xlabel("# Epoca")
    plt.ylabel("Magnitud de perdida")
    plt.plot(historial.history['loss'])
    plt.show()
    
    ##### ARRAY QUE MANDA LOS LOSS historial.history['loss']
    
    loss_curve=historial.history["loss"]
    accuracy_curve = historial.history['accuracy']
    plt.plot(accuracy_curve, label='precision')
    plt.plot(loss_curve,label='perdida')
    plt.legend(loc='lower left')
    plt.legend()
    plt.title('Resultado del Entrenamiento')
    plt.show()

except Exception as ex:
    print(ex)
#-----------------------------------------------------#
