# Customer Churn ML - Proyecto Integrador

Sistema de prediccion de abandono de clientes (churn) para una empresa de telecomunicaciones. Proyecto Integrador de la materia Laboratorio de Mineria de Datos, ISTEA.

**Alcance de esta entrega (Entrega 1 - Primer Parcial):** por indicacion del profesor en clase, esta entrega cubre hasta DVC/DagsHub. MLflow y Model Registry quedan para una etapa posterior de la cursada.

## Integrantes del grupo

- Aragusuku Pablo Ariel
- Silva Tobias
- Piedrabuena Giuliana
- Grossi Juan Pablo
- Di Renzo Tomas
- Lopez Maria

## Problema de negocio

Se busca estimar la probabilidad de que un cliente abandone el servicio (Churn: Yes/No), a partir de sus datos de contrato, servicios contratados y facturacion. El identificador de cliente no se utiliza como variable predictora.

## Dataset

Dataset de Customer Churn de telecomunicaciones (7043 clientes, 21 columnas). Variable objetivo: `Churn`, con distribucion desbalanceada (73.6% No / 26.4% Yes).

El dataset esta versionado con DVC, con remote configurado en DagsHub. No se encuentra versionado directamente en Git.

## Estructura del proyecto

- `data/raw/` - Dataset, versionado con DVC
- `notebooks/` - Exploracion y analisis (EDA)
- `src/config.py` - Constantes del proyecto
- `src/data/` - Carga y particion de datos
- `src/features/` - Pipeline de preprocessing
- `src/training/` - Script de entrenamiento
- `models/` - Modelo entrenado, no versionado en Git (carpeta vacia en el repositorio: hay que crearla manualmente antes de entrenar, ver Instalacion)
- `requirements.txt` - Dependencias del proyecto
- `README.md` - Este archivo

## Instalacion

1. Clonar el repositorio:

git clone https://github.com/jpgrossi/customer-churn-ml.git
cd customer-churn-ml

Importante: clonar en una ruta corta (por ejemplo `C:\Proyectos\customer-churn-ml`), evitando carpetas de OneDrive o rutas muy largas. Algunas dependencias (jupyterlab) generan archivos con nombres largos que pueden superar el limite de ruta de Windows y hacer fallar la instalacion.

2. Crear y activar el entorno virtual:

python -m venv .venv
.venv\Scripts\activate

3. Instalar dependencias:

pip install -r requirements.txt

4. Crear la carpeta de modelos (Git no versiona carpetas vacias, por lo que no se clona automaticamente):

mkdir models

5. Recuperar el dataset desde DVC:

dvc pull

(Requiere tener configurado el remote de DagsHub con credenciales validas, ver seccion "DVC" mas abajo. Si `dvc pull` no descarga el archivo, ver la seccion "Solucion de problemas".)

## Exploracion de datos (EDA)

El analisis exploratorio se encuentra en `notebooks/01_eda.ipynb`. Incluye dimensiones del dataset, tipos de datos, valores faltantes (26 nulos en `TotalCharges`), distribucion del target y cardinalidad de las variables categoricas.

## Pipeline de preprocessing

El preprocessing esta implementado con un `ColumnTransformer` de scikit-learn (`src/features/build_pipeline.py`):
- Variables numericas (`tenure`, `MonthlyCharges`, `TotalCharges`): imputacion por mediana + escalado estandar.
- Variables categoricas: imputacion por valor mas frecuente + one-hot encoding.

El mismo pipeline se usa en entrenamiento, garantizando que los datos se transformen de forma consistente.

## Modelos evaluados

Se compararon tres alternativas, todas con el mismo pipeline de preprocessing:

| Modelo | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Baseline (DummyClassifier) | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.664 | 0.452 | 0.538 | 0.812 |
| Random Forest | 0.631 | 0.414 | 0.500 | 0.794 |

Metricas calculadas sobre el conjunto de test (20% del dataset, particion estratificada, `random_state=42`), para la clase positiva "Yes" (cliente que abandona).

### Justificacion de la metrica y del modelo elegido

Se prioriza el Recall de la clase "Yes" sobre Accuracy o Precision. Un falso negativo (predecir que un cliente se queda cuando en realidad abandona) tiene mayor costo de negocio que un falso positivo: implica no poder ofrecerle una accion de retencion a un cliente que realmente se iba a ir, perdiendolo sin intentar retenerlo. Un falso positivo, en cambio, solo implica una accion de retencion de mas sobre un cliente que de todas formas se hubiera quedado.

Bajo este criterio, se selecciona **Logistic Regression** como modelo candidato: supera a Random Forest en las cuatro metricas, incluyendo Recall (0.452 vs 0.414) y ROC-AUC (0.812 vs 0.794).

**Limitacion conocida:** el Recall obtenido (0.452) indica que el modelo no detecta a mas de la mitad de los clientes que efectivamente abandonan. Queda como oportunidad de mejora para etapas futuras (ajuste de threshold, balanceo de clases, u otras tecnicas).

## Entrenamiento

El entrenamiento se ejecuta desde consola, sin depender de la ejecucion manual de un notebook:

python -m src.training.train

El script carga los datos, realiza la particion train/test, entrena el pipeline con Logistic Regression, calcula las metricas sobre el conjunto de test, y guarda el modelo entrenado en `models/churn_pipeline.joblib`. La carpeta `models/` debe existir previamente (ver Instalacion, paso 4).

Metricas esperadas (verificadas en mas de una maquina, confirmando reproducibilidad): Precision 0.664, Recall 0.452, F1 0.538, ROC-AUC 0.812.

## DVC

El dataset esta versionado con DVC, con remote configurado en DagsHub (`https://dagshub.com/jpgrossi/customer-churn-ml`). Para configurar las credenciales localmente:

dvc remote modify origin --local auth basic
dvc remote modify origin --local user usuario
dvc remote modify origin --local password token

El token se genera desde la configuracion de cuenta en DagsHub (Settings > Tokens) y nunca se sube al repositorio.

## Solucion de problemas

### `dvc pull` decia "Everything is up to date" sin descargar el dataset (resuelto)

**Causa:** el `.gitignore` tenia la regla `data/raw/*` junto con la excepcion `!data/raw/*.dvc`. Git interpreta bien ese par de reglas, pero DVC usa otra libreria (dulwich) para leer el `.gitignore` y considera ignorada la carpeta `data/raw/`. Por eso DVC no entraba a la carpeta, no encontraba `customer_churn_historical.csv.dvc` y `dvc pull`, `dvc push`, `dvc fetch` y `dvc checkout` terminaban sin hacer nada. El problema no dependia de Windows ni de la instalacion. El push indicando el archivo `.dvc` si funcionaba porque no necesita buscarlo en las carpetas.

**Solucion:** el `.gitignore` ahora ignora solo los CSV con `data/raw/*.csv`. No volver a usar `data/raw/*` en este repositorio.

Para comprobar que DVC detecta el dataset:

dvc status

No debe aparecer el mensaje "There are no data or pipelines tracked in this project yet".

### `dvc pull` informa "Some of the cache files do not exist neither locally nor on remote"

DVC muestra este mensaje tanto si faltan las credenciales de DagsHub (ver seccion "DVC") como si el archivo no esta en el remote. Primero configurar las credenciales y repetir `dvc pull`. Si el mensaje continua, subir el dataset con `dvc push` desde una maquina que lo tenga en `data/raw/`.

### Error de instalacion por rutas largas en Windows (OSError relacionado a jupyterlab/galata)

Clonar el proyecto en una ruta corta (por ejemplo `C:\Proyectos\customer-churn-ml`), evitando carpetas de OneDrive, espacios o nombres largos en el path completo.
