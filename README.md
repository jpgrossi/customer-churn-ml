# Customer Churn ML - Proyecto Integrador

Sistema de prediccion de abandono de clientes (churn) para una empresa de telecomunicaciones. Proyecto Integrador de la materia Laboratorio de Mineria de Datos, ISTEA.

## Integrantes del grupo

- Aragusuku Pablo Ariel
- Silva Tobias
- Piedrabuena Giuliana
- Grossi Juan Pablo
- Di Renzo Tomas

## Alcance

- **Entrega 1 (tag `entrega-1`):** proyecto reproducible con Git/GitHub, pipeline de scikit-learn y dataset versionado con DVC/DagsHub. Por indicacion del profesor, MLflow y Model Registry quedaron para la etapa siguiente.
- **Entrega 2 (tag `entrega-2`):** experimentos en MLflow, modelo registrado en Model Registry, API de inferencia con FastAPI/Pydantic y tests con pytest. Por indicacion del profesor, se entrega sin Docker.

## Problema de negocio

Estimar la probabilidad de que un cliente abandone el servicio (`Churn`: Yes/No) a partir de sus datos de contrato, servicios y facturacion, y devolver un nivel de riesgo (LOW / MEDIUM / HIGH) que pueda usar otro sistema, por ejemplo un CRM o una campana de retencion. El identificador de cliente no se utiliza como variable predictora.

## Dataset

Customer Churn de telecomunicaciones: 7043 clientes, 21 columnas. Variable objetivo `Churn`, con distribucion desbalanceada (73.6% No / 26.4% Yes).

El dataset esta versionado con DVC, con remote configurado en DagsHub. No se encuentra versionado directamente en Git.

## Estructura del proyecto

- `data/raw/` - Dataset, versionado con DVC
- `notebooks/` - Exploracion y analisis (EDA)
- `src/config.py` - Constantes: columnas, semillas, nombre del modelo y umbrales de riesgo
- `src/data/` - Carga y particion de datos
- `src/features/` - Pipeline de preprocessing
- `src/training/train.py` - Entrena los 6 modelos y los registra como runs en MLflow
- `src/training/register_model.py` - Registra el modelo elegido en el Model Registry
- `scripts/risk_thresholds.py` - Analisis que justifica los umbrales de riesgo
- `app/` - API de inferencia (FastAPI)
- `tests/` - Tests con pytest
- `.env.example` - Plantilla de variables de entorno, sin secretos
- `requirements.txt` - Dependencias del proyecto

## Instalacion

Requisitos: Python 3.12 y Git.

1. Clonar el repositorio:

```
git clone https://github.com/jpgrossi/customer-churn-ml.git
cd customer-churn-ml
```

Importante: clonar en una ruta corta (por ejemplo `C:\Proyectos\customer-churn-ml`), evitando carpetas de OneDrive o rutas muy largas, que pueden superar el limite de ruta de Windows y hacer fallar la instalacion.

2. Crear y activar el entorno virtual:

```
py -3.12 -m venv .venv
.venv\Scripts\activate
```

3. Instalar dependencias:

```
pip install -r requirements.txt
```

4. Configurar las credenciales de MLflow: copiar `.env.example` como `.env` y completar usuario y token de DagsHub. El token se genera en DagsHub (Settings > Tokens). El archivo `.env` esta en `.gitignore` y nunca se sube al repositorio.

5. Configurar DVC con las mismas credenciales y recuperar el dataset:

```
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <usuario>
dvc remote modify origin --local password <token>
dvc pull
```

Las credenciales se guardan en `.dvc/config.local`, que tampoco se sube al repositorio.

## Exploracion de datos (EDA)

El analisis exploratorio se encuentra en `notebooks/01_eda.ipynb`. Incluye dimensiones del dataset, tipos de datos, valores faltantes (26 nulos en `TotalCharges`), distribucion del target y cardinalidad de las variables categoricas.

## Pipeline de preprocessing

El preprocessing esta implementado con un `ColumnTransformer` de scikit-learn (`src/features/build_pipeline.py`):

- Variables numericas (`tenure`, `MonthlyCharges`, `TotalCharges`): imputacion por mediana + escalado estandar.
- Variables categoricas: imputacion por valor mas frecuente + one-hot encoding con `handle_unknown="ignore"`.

El preprocessing y el modelo se guardan juntos como un unico Pipeline, de modo que el entrenamiento y la API aplican exactamente las mismas transformaciones.

## Experimentos (MLflow)

Los experimentos se registran en el MLflow de DagsHub, experimento `churn-classification`: https://dagshub.com/jpgrossi/customer-churn-ml

El entrenamiento se ejecuta desde consola, sin depender de un notebook:

```
python -m src.training.train
```

Cada run registra parametros, metricas (accuracy, precision, recall, F1, ROC-AUC), la matriz de confusion, el pipeline completo y tags de trazabilidad: `git_sha` (commit del codigo) y `dataset_md5` (version del dataset segun el archivo DVC). Todos los runs usan la misma particion (80/20 estratificada, `random_state=42`), para que la comparacion sea justa.

| Run | Pregunta que responde | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| baseline (Dummy) | Cual es el piso minimo? | 0.000 | 0.000 | 0.000 | 0.500 |
| logreg | Referencia: modelo de la Entrega 1 | 0.664 | 0.452 | 0.538 | 0.812 |
| logreg_balanced | Compensar el desbalance mejora el recall? | 0.481 | **0.723** | 0.578 | **0.812** |
| rf | Referencia con arboles | 0.631 | 0.414 | 0.500 | 0.794 |
| rf_depth8 | Limitar la profundidad mejora? | 0.684 | 0.384 | 0.492 | 0.804 |
| rf_balanced | El balanceo tambien ayuda en arboles? | 0.493 | 0.710 | 0.581 | 0.804 |

Metricas calculadas sobre el conjunto de test, para la clase positiva "Yes" (cliente que abandona).

### Seleccion del modelo

Se prioriza el **Recall** de la clase "Yes" sobre Accuracy o Precision. Un falso negativo (predecir que un cliente se queda cuando en realidad abandona) tiene mayor costo de negocio que un falso positivo: implica perder al cliente sin intentar retenerlo. Un falso positivo solo implica una accion de retencion de mas sobre un cliente que igual se hubiera quedado.

En la Entrega 1 se eligio Logistic Regression, con la limitacion de que su recall (0.452) no detectaba a mas de la mitad de los clientes que abandonan. En esta entrega se ataco ese problema con `class_weight="balanced"`, que hace que el modelo penalice mas los errores sobre la clase minoritaria. El recall sube de 0.45 a 0.72 en regresion logistica y de 0.41 a 0.71 en random forest.

Entre las dos variantes balanceadas, la diferencia es de alrededor de 0.01 en cada metrica. Se elige **logreg_balanced** porque gana en la metrica principal (recall) y en ROC-AUC, y es el modelo mas simple, liviano y explicable.

Costo de la decision: frente al modelo de la Entrega 1, se detectan 72 de cada 100 clientes que abandonan (antes 45), y la precision baja de 0.66 a 0.48.

**Limitacion conocida:** la comparacion usa una unica particion de test. Para confirmar que la diferencia entre los modelos balanceados es estable habria que usar validacion cruzada.

## Model Registry

El modelo elegido esta registrado como **`CustomerChurnModel`, version 1, alias `candidate`**:

```
python -m src.training.register_model
```

La version tiene tags con el nombre del run de origen, `git_sha`, `dataset_md5`, metrica principal y clase positiva, y una descripcion con la justificacion. Desde la version registrada se puede reconstruir el run, el commit y la version de datos que la produjeron. La API carga el modelo por su alias, no por un archivo local.

Nota: el script crea una version nueva cada vez que se ejecuta. Se corre una sola vez por cada decision de registro.

## Niveles de riesgo

| Nivel | Probabilidad de churn | Clientes (test) | Abandono real |
|---|---|---|---|
| LOW | menor a 0.40 | 52% | 9.2% |
| MEDIUM | 0.40 a 0.70 | 26% | 32.2% |
| HIGH | 0.70 o mas | 21% | 61.3% |

Calculado con `python -m scripts.risk_thresholds`. Los umbrales separan grupos con riesgo realmente distinto: un cliente HIGH tiene mas del doble de probabilidad de irse que el promedio del dataset (26.4%), y el nivel forma una lista acotada para priorizar acciones de retencion. Los umbrales estan definidos en `src/config.py`.

`prediction` usa la regla del modelo (probabilidad >= 0.5), mientras que `risk_level` es la traduccion operativa en tres tramos. Por eso un cliente puede ser MEDIUM con `prediction` "No": no llega a ser churn segun el modelo, pero conviene vigilarlo.

## API de inferencia

Levantar la API (con el `.env` configurado, porque al arrancar descarga el modelo desde DagsHub):

```
uvicorn app.main:app --reload
```

Documentacion interactiva (Swagger): http://127.0.0.1:8000/docs

**GET /health** - Indica si el servicio esta listo y que version del modelo tiene cargada. Responde 503 si el modelo no pudo cargarse.

```json
{"status": "ok", "model_version": "1"}
```

**POST /predict** - Recibe las 19 caracteristicas de un cliente y devuelve la prediccion.

Ejemplo de request:

```json
{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
  "tenure": 5, "PhoneService": "Yes", "MultipleLines": "No",
  "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
  "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 470.0
}
```

Ejemplo de respuesta:

```json
{
  "prediction": "Yes",
  "probability": 0.971,
  "risk_level": "HIGH",
  "model_version": "1",
  "request_id": "d8e7d3d0-b497-459c-97f4-e140472c0817"
}
```

Validacion con Pydantic: categorias permitidas por columna, `tenure >= 0`, `MonthlyCharges > 0` y `TotalCharges >= 0` (puede venir vacio; el pipeline lo imputa). Un request invalido devuelve **422** indicando el campo y el motivo, sin llegar al modelo.

Estructura de `app/`: `main.py` (rutas HTTP), `schemas.py` (contratos de entrada y salida), `model_loader.py` (carga desde el Registry) e `inference.py` (prediccion y nivel de riesgo, independiente de HTTP). El modelo se carga una sola vez, al arrancar la API.

## Tests

```
pytest -v
```

19 tests en `tests/`:

- `test_inference.py`: carga del modelo, clases conocidas, contrato de la prediccion, `TotalCharges` vacio y bordes de los umbrales de riesgo.
- `test_api.py`: `/health`, `/predict` con un cliente valido, `request_id` unico por pedido, cinco tipos de valores invalidos y un campo faltante (todos deben devolver 422).

**Limitacion:** los tests que usan el modelo lo descargan de DagsHub, por lo que necesitan conexion a internet y el `.env` configurado.

## Decisiones tecnicas

- **MLflow 2 (2.22.5) en lugar de MLflow 3:** MLflow 3 guarda los modelos en una seccion nueva ("logged models") que el servidor de DagsHub no maneja, y el Model Registry no encontraba el modelo dentro del run. Con MLflow 2 el modelo se guarda dentro del run y el registro funciona.
- **Python 3.12:** version estable, compatible con todas las dependencias.
- **requirements.txt con dependencias directas:** solo los paquetes que usa el proyecto, con versiones fijas. Se quitaron paquetes exclusivos de Windows, para que la instalacion funcione tambien en Linux/Docker. Verificado instalando en un entorno limpio y corriendo los tests.

## Solucion de problemas

### dvc pull decia "Everything is up to date" sin descargar el dataset (resuelto)

Causa: el `.gitignore` tenia la regla `data/raw/*` junto con la excepcion `!data/raw/*.dvc`. Git interpreta bien ese par de reglas, pero DVC usa otra libreria (dulwich) para leer el `.gitignore` y considera ignorada la carpeta `data/raw/`. Por eso DVC no entraba a la carpeta, no encontraba `customer_churn_historical.csv.dvc`, y `dvc pull`, `dvc push`, `dvc fetch` y `dvc checkout` terminaban sin hacer nada. El problema no dependia de Windows ni de la instalacion. El push indicando el archivo `.dvc` si funcionaba porque no necesita buscarlo en las carpetas.

Solucion: el `.gitignore` ahora ignora solo los CSV con `data/raw/*.csv`. No volver a usar `data/raw/*` en este repositorio.

Para comprobar que DVC detecta el dataset:

```
dvc status
```

No debe aparecer el mensaje "There are no data or pipelines tracked in this project yet".

### dvc pull informa "Some of the cache files do not exist neither locally nor on remote"

DVC muestra este mensaje tanto si faltan las credenciales de DagsHub como si el archivo no esta en el remote. Primero configurar las credenciales y repetir `dvc pull`. Si el mensaje continua, subir el dataset con `dvc push` desde una maquina que lo tenga en `data/raw/`.

### Error de instalacion por rutas largas en Windows

Clonar el proyecto en una ruta corta (por ejemplo `C:\Proyectos\customer-churn-ml`), evitando carpetas de OneDrive, espacios o nombres largos en el path completo.

### Error 401 al conectar con MLflow

Revisar usuario y token en `.env`. El token de DagsHub tiene 40 caracteres; un espacio de mas al copiarlo provoca el error.
