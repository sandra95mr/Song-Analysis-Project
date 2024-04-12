import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import re
from tensorflow.keras.callbacks import Callback, TensorBoard
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.regularizers import l2



#tensorboard --logdir='C:/Users/sandr/OneDrive/Escritorio/train'

def scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        return lr * 0.9
    else:
        return lr

# Define the plot_confusion_matrix function
def plot_confusion_matrix(model, x_val, y_val):
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_val_classes = np.argmax(y_val, axis=1)

    cm = confusion_matrix(y_val_classes, y_pred_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_val_classes),
                yticklabels=np.unique(y_val_classes))
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

class CustomCallback(Callback):
    def __init__(self, x_val, y_val):
        self.x_val = x_val
        self.y_val = y_val

    def on_epoch_end(self, epoch, logs=None):
        print("Custom callback executed at epoch", epoch)

        if self.x_val is not None and self.y_val is not None:
            print("Shape of x_val:", self.x_val.shape)
            print("Shape of y_val:", self.y_val.shape)
            print("Generating confusion matrix...")
            plot_confusion_matrix(self.model, self.x_val, self.y_val)

            # Compute and print classification report
            y_pred = np.argmax(self.model.predict(self.x_val), axis=1)
            y_true = np.argmax(self.y_val, axis=1)
            report = classification_report(y_true, y_pred, zero_division=1)
            print("Classification Report:\n", report)
        else:
            print("Validation data not found.")


# Configurar la visualización de pandas para que se vean en pantalla toda la columna
pd.set_option("display.max_rows", None)  # Mostrar todas las filas
pd.set_option("display.max_columns", None)  # Mostrar todas las columnas

# Especifica la ruta de tu archivo Excel
ruta_del_archivo = 'Datos.xlsx'
ruta_del_archivo2 = 'Datos_Entrenamiento.xlsx'
ruta_del_archivo3 = 'Prueba.xlsx'

tensorboard_callback = TensorBoard(log_dir='C:/Users/sandr/OneDrive/Escritorio', histogram_freq=1)

# Lee la hoja de cálculo que contiene tus datos
df = pd.read_excel(ruta_del_archivo, header=None, names=['Palabra', 'Etiqueta'])

# Lee la hoja de cálculo que contiene tus datos
df2 = pd.read_excel(ruta_del_archivo2, header=None, names=['Texto', 'Vector'])

# Lee la hoja de cálculo que contiene tus datos
df3 = pd.read_excel(ruta_del_archivo3, header=None, names=['Cancion'])

# Preprocesamiento de datos

# Si la columna "Palabra" no es de tipo cadena, conviértela y luego a minúsculas
df['Palabra'] = df['Palabra'].astype(str)
df['Palabra'] = df['Palabra'].str.lower()

# Si la columna "Texto" no es de tipo cadena, conviértela y luego a minúsculas
df2['Texto'] = df2['Texto'].astype(str)
df2['Texto'] = df2['Texto'].str.lower()

# Quitar comas, paréntesis y guiones
caracteres_a_quitar = [',', '(', ')', '-', ':']
for char in caracteres_a_quitar:
    df2['Texto'] = df2['Texto'].str.replace(char, '')

# Si la columna "Palabra" no es de tipo cadena, conviértela y luego a minúsculas
df3['Cancion'] = df3['Cancion'].astype(str)
df3['Cancion'] = df3['Cancion'].str.lower()

for char in caracteres_a_quitar:
    df3['Cancion'] = df3['Cancion'].str.replace(char, '')

# Convertir 'Etiqueta' a numérico
df['Etiqueta'] = pd.to_numeric(df['Etiqueta'], errors='coerce')


# Eliminar filas con valores nulos en la columna 'Vector'
df2.dropna(subset=['Vector'], inplace=True)

# Quitar el decimal de los valores restantes
df2['Vector'] = df2['Vector'].apply(lambda x: re.sub(r'\.\d+', '', str(x)))

# Convertir los valores restantes a un vector de enteros
df2['Vector'] = df2['Vector'].apply(lambda x: [int(digit) for digit in re.findall(r'\d', str(x))])

# Crear listas

# Obtén las frases y las etiquetas
frases = df['Palabra'].tolist()
etiquetas = df['Etiqueta'].tolist()
textos=df2['Texto'].tolist()
vector= df2['Vector'].tolist()
cancion=df3['Cancion'].tolist()

# Tokenizar las palabras usando embedding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(frases)

# Obtener secuencias de palabras
secuencias = tokenizer.texts_to_sequences(frases)

# Calcular la longitud de la secuencia más larga
longitud_maxima = max(len(secuencia) for secuencia in secuencias)

# Verificar el mapeo numérico con el siguiente código
mapeo_etiquetas = tokenizer.word_index
#print("Mapeo numérico de etiquetas:", mapeo_etiquetas) #mi etiqueta 8 en token es un 1, mi 7 en token un 2... mi etiqueta sexo es 41

# Obtén secuencias de palabras para textos
secuencias_texto = tokenizer.texts_to_sequences(textos)

# Calcular la longitud de la secuencia más larga
longitud_maxima_texto = max(len(secuencia) for secuencia in secuencias_texto)

# Obtener la longitud máxima entre texto y palabras
longitud_maxima_total = max(longitud_maxima_texto, longitud_maxima)

# Aplicar padding a las secuencias de texto
secuencias_texto_padded = pad_sequences(secuencias_texto, maxlen=longitud_maxima_total, padding='post')

#Aplicar padding a las secuencias de palabras
secuencias_padded= pad_sequences(secuencias, maxlen=longitud_maxima_total, padding='post')

# Crear un DataFrame para mostrar las secuencias con embedding
df_secuencias = pd.DataFrame(secuencias_padded)

# Obtén secuencias de palabras para textos
secuencias_cancion = tokenizer.texts_to_sequences(cancion)


# Calcular las longitudes después del padding, contando ceros y no ceros para comprobar
longitudes = df_secuencias.apply(lambda row: len(row), axis=1)


num_etiquetas_unicas = df['Etiqueta'].nunique() #Utilizar la función nunique() en la columna que contiene las etiquetas
#para sacar el numero de salidas que espero de la red


# Inicializar una lista vacía para almacenar las listas de etiquetas
etiquetas_lista = []

# Iterar sobre cada número de etiqueta y crear su vector binario
for eti in etiquetas:
    vector_binario = [0] * num_etiquetas_unicas
    vector_binario[eti - 1] = 1
    etiquetas_lista.append(vector_binario)

# Crear una lista de vectores binarios
vectores_binarios = []

# Iterar sobre cada lista de números en vector y crear su vector binario
for lista_numeros in vector:
    # Inicializar un vector binario para esta lista
    vector_binario = [0] * 8

    # Establecer a 1 las posiciones correspondientes a los números en la lista
    for numero in lista_numeros:
        vector_binario[numero - 1] = 1

    vectores_binarios.append(vector_binario)


#Arquitectura del modelo

vocab_size = len(tokenizer.word_index) + 1 #Este es el tamaño del vocabulario,
#es decir, el número total de palabras únicas en tus datos. Lo puedes obtener a partir del atributo word_index del
#Tokenizer que has utilizado para tokenizar tus palabras.

embedding_dim=100 #Esta es la dimensión del espacio de embedding.
#Cuántos números representan cada palabra. Por lo general, se elige experimentando.
#Valores comunes: 50, 100, 200. Si tienes muchas palabras y datos, un embedding más grande podría ser útil.

max_length = df_secuencias.shape[1] # Este valor es la longitud máxima de las secuencias después del padding.
#La longitud máxima de los DataFrames

#num_etiquetas_unicas = df_concatenado['Etiqueta'].nunique() #Utilizar la función nunique() en la columna que contiene las etiquetas
#para sacar el numero de salidas que espero de la red


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length), #primera capa convierte las secuencias de palabras en vectores densos llamados embeddings
    #input_dim -> palabras totales entre todas las tematicas : output_dim -> 100 uno de los valores comunes : input_length -> como nuestro max_length la cantidad de palabras que tiene la tematica que más palabras tiene
    tf.keras.layers.Conv1D(128, 5, activation='relu'),  #segunda capa que aplica un filtro a las secuencias de embedding para detectar patrones locales (128 es el número de patrones que buscara) (el filtro se desliza sobre ventanas de 5 palabras a la vez) (la funcion de activacion ReLU se aplica despues de la convolucion)
    tf.keras.layers.GlobalMaxPooling1D(),  #esta capa toma el valor maximo a lo largo de todas las dimensiones (palabras) para reducir la dimension de los datos y conservar caracteristicas relevantes. Ayuda a capturar los patrones más importantes de cada filtro
    tf.keras.layers.BatchNormalization(),
    # Capa densa para la clasificación
    tf.keras.layers.Dense(164, activation='relu',kernel_regularizer=l2(0.01)),
    tf.keras.layers.Dense(units=64, activation='relu'), #se utiliza comúnmente en capas ocultas de redes neuronales para introducir no linealidades en el modelo
    #64 indica el número de neuronas o unidades en esa capa. Cuantas más unidades tenga en una capa, más complejas pueden ser las representaciones
    tf.keras.layers.Dropout(0.5), # Dropout para reducir el sobreajuste
    #Apaga aleatoriamente un cierto porcentaje de unidades (en este caso, el 50% o 0.5) durante el entrenamiento.
    #Esto significa que, durante cada paso de entrenamiento, la mitad de las unidades se desactivan temporalmente
    tf.keras.layers.Dense(num_etiquetas_unicas,activation='sigmoid'), #esta cuarta capa se conecta con todas las unidades de la capa anterior y realiza una transformacion lineal seguida de una funcion de activacion ReLU (64 unidades en este caso pero se puede ajustar a tus necesidades)
    # Utilizo enfoque no supervisado para aprender patrones en los datos de entrada y no tengo datos de destino (etiquetas) o sea la capa de salida sobra.

])


#-----------------------------

#Compilación

#Optimizer: ajusta los pesos del modelo durante el entrenamiento para minimizar la función de pérdida. SGD, RMSprop, Adagrad son otros optimizadores

#optimizer = Adam(learning_rate=0.001)  # Prueba con diferentes tasas de aprendizaje


optimizer = Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, #es una buena opción para comenzar. Adapta automáticamente las tasas de aprendizaje durante el entrenamiento
              loss='categorical_crossentropy', #Esta función de pérdida es adecuada para problemas de clasificación multiclase donde las etiquetas son enteros y representan clases mutuamente excluyentes
              metrics=['accuracy'])  #si ha codificado su objetivo en caliente para que tenga forma 2D (n_samples, n_class), puede usarcategorical_crossentropy

            #Accuracy es una métrica común para problemas de clasificación.
            #mejora la precisión del modelo en el conjunto de entrenamiento.
            #Puedes agregar más métricas según sea necesario, como ['accuracy', 'precision', 'recall'].

#Loss Function: funcion de perdida, para medir cuan bien se está desempeñando el modelo


#Metrics evaluan el rendimiento del modelo durante el entrenamiento y la evaluacion (otras metricas: precision ->accuracy. recuperacion -> recall. precision -> precision. el valor F1 ->f1. el area bajo la curva ROC -> roc_auc...)
    #Si la precisión es importante, la métrica 'accuracy'. Si lo es más la detección de casos positivos'recall'. Se pueden usar una o varias metricas segun necesidad.
#-----------------------------

model.summary()  #da informacion

input_shape = model.layers[0].input_shape
print("Forma de entrada esperada:", input_shape)

#-----------------------------

#Entrenamiento

# Definir tus datos de entrada y salida
x_train = secuencias_padded  # secuencias de texto
y_train = etiquetas_lista  # secuencias de etiquetas

x_val= secuencias_texto_padded
y_val= vectores_binarios


# Supongamos que y_train es una lista
y_train = np.array(y_train)

# Asegúrate de que x_train también sea una matriz NumPy si aún no lo es
x_train = np.array(x_train)

# Supongamos que y_val es una lista
y_val = np.array(y_val)

# Asegúrate de que x_val también sea una matriz NumPy si aún no lo es
x_val = np.array(x_val)

num_epochs = 20

# Entrenar el modelo

custom_callback = CustomCallback(x_val, y_val)

lr_scheduler = LearningRateScheduler(scheduler)
callbacks = [lr_scheduler]

callbacks=[tensorboard_callback, custom_callback]

history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=32, validation_data=(x_val, y_val), callbacks=[tensorboard_callback, lr_scheduler], shuffle=True)

#número de veces que el modelo verá todo el conjunto de datos de entrenamiento.
#más épocas y el modelo aprende más de los datos, pero también podría aumentar el riesgo de sobreajuste

#El tamaño del lote se refiere al número de muestras que se utilizan en cada iteración de entrenamiento.
#más grande puede acelerar el entrenamiento, pero también puede requerir más memoria.
#más pequeño podría ser más lento pero podría ayudar con la convergencia del modelo.
#32 y 64 tipicos de muestras pequeñas

#Durante el entrenamiento, es útil reservar una parte de tus datos de entrenamiento
#como conjunto de validación para evaluar el rendimiento del modelo en datos no vistos.
#El parámetro validation_split en Keras te permite especificar la proporción de tus datos
# de entrenamiento que se deben usar como conjunto de validación

#shuffle=True ->  Esto significa que las muestras (pares de palabras y etiquetas)
# se presentarán en un orden diferente en cada época, lo que ayuda al modelo a aprender
# patrones de manera más robusta y a generalizar mejor a datos no vistos.


# Graficar la pérdida en entrenamiento y validación
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida en entrenamiento y validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.legend()
plt.show()

# Graficar la precisión en entrenamiento y validación
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión en entrenamiento y validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.legend()
plt.show()


parrafos = []

# Iterar sobre las secuencias y agregar cada fragmento al array
for secuencia in secuencias_cancion:
    if len(secuencia) <= longitud_maxima_total:
        # Si la secuencia es más corta que la longitud máxima, agrégala directamente
        parrafos.append(secuencia)
    else:
        for i in range(0, len(secuencia), longitud_maxima_total):
            # Obtener un fragmento de longitud máxima o lo que queda si es menor
            fragmento = secuencia[i:i + longitud_maxima_total]
            # Añadir el fragmento a la lista
            parrafos.append(fragmento)

# Iterar sobre los fragmentos y aplicar relleno
parrafos = pad_sequences(parrafos, maxlen=longitud_maxima_total, padding='post', value=0)

#for i in parrafos:
   # print(i)


# Obtener predicciones para las frases
predicciones_totales = []

# Comprobar que también sea una matriz NumPy si aún no lo es
muestra = np.array(parrafos)

for i in range(muestra.shape[0]):
    prediccion = model.predict(np.expand_dims(muestra[i], axis=0))

    # Binarizar las predicciones usando un umbral de 0.5
    predicciones_binarizadas = (prediccion > 0.5).astype(int)
    
    predicciones_totales.append(predicciones_binarizadas)

# Obtener el resultado final combinando las predicciones
resultado_final = np.max(predicciones_totales, axis=0)


# Imprimir el resultado final
print("Resultado final:", resultado_final)



'''
# Obtén predicciones en el conjunto de validación
predicciones_validacion = model.predict(x_val)

# Convierte las predicciones a etiquetas binarias (0 o 1)
predicciones_binarias = (predicciones_validacion > 0.5).astype(int)

# Crear un DataFrame con los resultados
df_resultados = pd.DataFrame({
    'Texto': textos[:len(x_val)],  # Muestra original del conjunto de validación
    'Prediccion': predicciones_binarias.tolist(),
    'Etiqueta Real': vectores_binarios[:len(x_val)]  # Etiquetas reales del conjunto de validación
})

# Filtra ejemplos donde el modelo cometió errores
errores = df_resultados[df_resultados['Prediccion'] != df_resultados['Etiqueta Real']]


for index, row in errores.iterrows():
    print(f"\nTexto: {row['Texto']}")
    print(f"Predicción: {row['Prediccion']}")
    print(f"Etiqueta Real: {row['Etiqueta Real']}")

'''