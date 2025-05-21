# Este script utiliza un modelo de red neuronal previamente entrenado para predecir la readmisión a la UCI (Unidad de Cuidados Intensivos) a partir de un conjunto de datos.
# Pasos principales:
# 1. Define la arquitectura del modelo para que coincida con los pesos guardados.
# 2. Carga los pesos entrenados desde un archivo .h5.
# 3. Carga los datos a predecir desde un archivo CSV.
# 4. Separa las variables predictoras (X) y la variable objetivo (y).
# 5. Realiza la predicción con el modelo.
# 6. Convierte la salida del modelo a formato binario (0 o 1) si es necesario.
# 7. Calcula y muestra métricas de desempeño (accuracy y reporte de clasificación).
# Este flujo permite evaluar el desempeño del modelo sobre nuevos datos y entender su capacidad para predecir la readmisión a la UCI.

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Definir la arquitectura del modelo según los pesos
model = Sequential([
    Dense(16, input_shape=(24,), activation='relu', name='dense'),
    Dense(16, activation='relu', name='dense_1'),
    Dense(256, activation='relu', name='dense_2'),
    Dense(1, activation='sigmoid', name='dense_3')
])

# Cargar los pesos
model.load_weights('.\\nn\\pesos_red_neuronal_84_44.h5')

# Cargar los datos
csv_path = '.\\nn\\predecir_readmision_uci.csv'
df = pd.read_csv(csv_path)

# Separar X e y
if 'Readmission to the ICU' not in df.columns:
    raise ValueError('La columna objetivo "Readmission to the ICU" no está en el archivo CSV.')
X = df.drop(columns=['Readmission to the ICU'])
y_true = df['Readmission to the ICU']

# Normalizar o escalar si es necesario (aquí se asume que los datos ya están listos para el modelo)

# Predecir
preds = model.predict(X)

# Si la salida es continua, convertir a binaria
if preds.shape[1] == 1 or len(preds.shape) == 1:
    y_pred = (preds > 0.5).astype(int).flatten()
else:
    y_pred = np.argmax(preds, axis=1)

# Validar
print('Accuracy:', accuracy_score(y_true, y_pred))
print('Reporte de clasificación:')
print(classification_report(y_true, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Gráfica de barras del reporte de clasificación
report = classification_report(y_true, y_pred, output_dict=True)
metrics = ['precision', 'recall', 'f1-score']
classes = [str(c) for c in report.keys() if c.isdigit()]
report_df = pd.DataFrame(report).T.loc[classes, metrics]
report_df.plot(kind='bar')
plt.title('Reporte de clasificación por clase')
plt.ylabel('Score')
plt.ylim(0,1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.show()