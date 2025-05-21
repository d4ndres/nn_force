import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras import backend as K

# 1. Cargar el archivo Excel
df = pd.read_excel("predecir_readmision_uci.xlsx")

# 2. Preparar X e y
X = df.drop(columns=["Readmission to the ICU"])
y = df["Readmission to the ICU"]

# 3. Eliminar NaNs
X = X.dropna()
y = y[X.index]

best_acc = 0
best_weights = None

for i in range(100):
    print(f"\nIteración {i+1}/100")
    # División aleatoria en cada iteración
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    K.clear_session()
    # Parámetros aleatorios para la red
    n_layers = random.randint(2, 6)  # entre 2 y 6 capas ocultas
    units_options = [16, 32, 64, 128, 256]
    dropout_options = [0, 0.1, 0.2, 0.3, 0.4]
    model = Sequential()
    # Primera capa (con input_shape)
    model.add(Dense(random.choice(units_options), activation='relu', input_shape=(X_train.shape[1],)))
    if random.random() < 0.5:
        model.add(Dropout(random.choice(dropout_options)))
    # Capas ocultas intermedias
    for _ in range(n_layers-1):
        model.add(Dense(random.choice(units_options), activation='relu'))
        if random.random() < 0.5:
            model.add(Dropout(random.choice(dropout_options)))
    # Capa de salida
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(
        X_train, y_train,
        epochs=random.randint(50, 200),  # número de épocas aleatorio
        batch_size=random.choice([4, 8, 16, 32]),  # batch size aleatorio
        validation_split=0.2,
        verbose=0
    )
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Precisión de la red neuronal: {accuracy * 100:.2f}%")
    if accuracy > 0.5:
        precision_str = f"{accuracy * 100:.2f}".replace('.', '_')
        model.save_weights(f"pesos_red_neuronal_{precision_str}.h5")
        print(f"Pesos guardados como: pesos_red_neuronal_{precision_str}.h5")
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = model.get_weights()

print(f"\nMejor precisión obtenida: {best_acc * 100:.2f}%")
if best_weights is not None:
    model.set_weights(best_weights)
    precision_str = f"{best_acc * 100:.2f}".replace('.', '_')
    model.save_weights(f"mejores_pesos_red_neuronal_{precision_str}.h5")
    print(f"Mejores pesos guardados como: mejores_pesos_red_neuronal_{precision_str}.h5")
