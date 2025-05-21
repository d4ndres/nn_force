# Proyecto: Predicción de Readmisión a la UCI con Red Neuronal

## Objetivo
Este proyecto contiene scripts en Python para explorar archivos HDF5, convertir archivos de Excel a CSV y utilizar una red neuronal previamente entrenada para predecir la readmisión de pacientes a la Unidad de Cuidados Intensivos (UCI). El flujo de trabajo permite analizar datos médicos y evaluar el desempeño del modelo de predicción.

### Descripción de los scripts

- **016.py**: Explora la estructura y los atributos de un archivo HDF5, mostrando los grupos, datasets y sus características.
- **017.py**: Convierte archivos de Excel (.xlsx) a formato CSV, facilitando el procesamiento de datos tabulares.
- **018.py**: Utiliza un modelo de red neuronal (cuyos pesos están en un archivo .h5) para predecir la readmisión a la UCI a partir de un archivo CSV. Incluye evaluación del modelo y visualización de métricas.

## Instalación de dependencias

1. Asegúrate de tener Python 3.7 o superior instalado.
2. Instala las dependencias necesarias ejecutando en la terminal:

```
pip install -r requirements.txt
```

## Uso de los scripts

1. **Explorar archivo HDF5**:
   - Edita y ejecuta `016.py` para inspeccionar la estructura de un archivo `.h5`.

2. **Convertir Excel a CSV**:
   - Ejecuta `017.py` para convertir un archivo `.xlsx` a `.csv`.

3. **Predecir readmisión a la UCI**:
   - Asegúrate de tener el archivo de pesos y el archivo CSV de datos.
   - Ejecuta `018.py` para realizar la predicción y visualizar los resultados.

## Notas
- Los archivos de pesos y datos deben estar en la carpeta `nn/` o ajusta las rutas en los scripts según corresponda.
- El archivo `requirements.txt` contiene todas las librerías necesarias para ejecutar los scripts.
