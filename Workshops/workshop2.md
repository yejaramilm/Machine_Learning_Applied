# Workshop 2 — Red neuronal profunda desde cero: resistencia del concreto

**Curso:** ST1613 Applied Machine Learning
**Competencia:** [Concrete Strength Regression (Kaggle)](https://www.kaggle.com/competitions/concrete-strength-regression)
**Framework:** PyTorch

---

## 1. Objetivo

Construir, entrenar y evaluar una **red neuronal profunda fully connected (MLP)** implementada **desde cero en PyTorch** para predecir la **resistencia a la compresión del concreto** a partir de su composición y edad.

El propósito del taller es que integren en un flujo de trabajo completo todo lo visto hasta ahora en el curso:

- Análisis exploratorio de datos (EDA)
- Limpieza y preprocesamiento
- Escalamiento de variables
- Codificación de variables categóricas (si aplica)
- Partición de datos y prevención de *data leakage*
- Diseño, entrenamiento y validación de una red neuronal
- Generación de predicciones para Kaggle

> **"Desde cero"** significa que ustedes definen la arquitectura (`nn.Module` / `nn.Sequential`), el `Dataset`, los `DataLoader`, la función de pérdida, el optimizador y los ciclos de entrenamiento y validación. **No** se permite usar modelos preentrenados ni librerías de AutoML, ni reemplazar la red por modelos de `scikit-learn` (aunque sí pueden usar `scikit-learn` para preprocesamiento y como *baseline* de comparación).

---

## 2. Material de referencia

Usen como punto de partida los notebooks vistos en clase:

| Notebook | Qué reutilizar |
|---|---|
| [01_mnist.ipynb](01_mnist.ipynb) | Estructura de un modelo `nn.Sequential` con capas `nn.Linear` + `nn.ReLU`, uso del optimizador `Adam`, y las funciones `train()` y `validate()`. |
| [02_asl.ipynb](02_asl.ipynb) | Carga de datos **tabulares** con `pandas`, clase `MyDataset(Dataset)` con `__init__`, `__getitem__` y `__len__`, construcción de `DataLoader`, y ciclo de entrenamiento por épocas. |

### ⚠️ Diferencias clave: estos ejemplos son de **clasificación**, este taller es de **regresión**

Al adaptar el código deben cambiar, como mínimo:

| Elemento | Clasificación (MNIST / ASL) | Regresión (este taller) |
|---|---|---|
| Capa de salida | `nn.Linear(..., n_clases)` | `nn.Linear(..., 1)` **sin activación** |
| Función de pérdida | `nn.CrossEntropyLoss()` | `nn.MSELoss()` (o `nn.L1Loss()`, `nn.HuberLoss()`) |
| Tipo/forma del target | `long`, forma `(N,)` | `float32`, forma `(N, 1)` |
| Métrica | `get_batch_accuracy` | **MSE** (métrica oficial), además de RMSE, MAE y R² (reemplacen la función de accuracy) |

---

## 3. Datos

Descarguen los archivos desde la pestaña **Data** de la competencia (deben aceptar las reglas de la competencia en Kaggle):

- `train.csv` — datos con la variable objetivo (resistencia del concreto).
- `test.csv` — datos sin la variable objetivo, sobre los cuales deben predecir.
- `sample_submission.csv` — formato exacto del archivo que deben subir.

### Descripción del dataset (*Concrete Compressive Strength*)

- **Instancias:** 721
- **Atributos:** 9 (8 predictores cuantitativos + 1 variable objetivo cuantitativa)
- **Valores faltantes:** ninguno, según la documentación (verifíquenlo de todas formas en el EDA)

| Variable | Tipo | Unidad | Descripción |
|---|---|---|---|
| Cement (componente 1) | Entrada | kg/m³ de mezcla | Cantidad de cemento |
| Blast Furnace Slag (componente 2) | Entrada | kg/m³ de mezcla | Cantidad de escoria de alto horno |
| Fly Ash (componente 3) | Entrada | kg/m³ de mezcla | Cantidad de ceniza volante |
| Water (componente 4) | Entrada | kg/m³ de mezcla | Cantidad de agua |
| Superplasticizer (componente 5) | Entrada | kg/m³ de mezcla | Cantidad de superplastificante |
| Coarse Aggregate (componente 6) | Entrada | kg/m³ de mezcla | Cantidad de agregado grueso |
| Fine Aggregate (componente 7) | Entrada | kg/m³ de mezcla | Cantidad de agregado fino |
| Age | Entrada | Días (1–365) | Edad del concreto al momento del ensayo |
| **Concrete Compressive Strength** | **Objetivo** | **MPa** | **Resistencia a la compresión** |

> Los nombres exactos de las columnas en los CSV pueden diferir de los de esta tabla; verifíquenlos con `df.columns`.

### Métrica de la competencia

La métrica oficial es el **Error Cuadrático Medio (MSE)**: entre menor, mejor.

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Úsenla como **métrica principal** en su validación, calculada en la escala original del target (MPa²). Como `nn.MSELoss()` optimiza justamente esta métrica, es la elección natural como función de pérdida.

### Observaciones importantes sobre este dataset

- **Todas las variables son numéricas:** no hay variables categóricas, así que **no se requiere encoding**. Indíquenlo y justifíquenlo en el notebook.
- **Es un dataset pequeño (721 filas):** una red profunda puede sobreajustarse con facilidad. La regularización, el early stopping y una validación cuidadosa (idealmente **validación cruzada K-Fold**) son especialmente importantes.
- **Las escalas son muy distintas:** algunas variables van de 0 a ~1000 kg/m³ y otras (Superplasticizer) apenas a unas decenas. Por eso el escalamiento es indispensable.
- **Varios componentes tienen muchos ceros** (Blast Furnace Slag, Fly Ash, Superplasticizer): no todas las mezclas los usan.
- **`Age` es muy asimétrica** y está concentrada en unos pocos valores (p. ej. 3, 7, 28, 90 días).

---

## 4. Actividades

### Parte 1 — EDA básico de reconocimiento

Antes de modelar, **conozcan los datos**. Como mínimo:

1. Dimensiones de `train` y `test`, tipos de datos (`df.info()`), y primeras filas.
2. Estadísticas descriptivas (`df.describe()`).
3. Valores faltantes y filas duplicadas.
4. Identificación de variables **numéricas** y **categóricas** (si las hay) y de columnas que no deben usarse como predictores (p. ej. `id`).
5. Distribución de la variable objetivo (histograma + boxplot). ¿Es asimétrica? ¿Hay outliers?
6. Distribución de cada predictor (histogramas). ¿Hay variables con muchos ceros o muy sesgadas?
7. Matriz de correlación y gráficos de dispersión de cada predictor vs. el objetivo.
8. Comparación de distribuciones entre `train` y `test` para detectar diferencias.

Cierren esta parte con **3 a 5 hallazgos** escritos en Markdown que justifiquen las decisiones de preprocesamiento que tomarán después.

### Parte 2 — Preprocesamiento

1. **Partición:** separen `train.csv` en entrenamiento y validación (p. ej. 80/20, con `random_state` fijo). **Toda transformación se ajusta (`fit`) solo con el conjunto de entrenamiento** y luego se aplica (`transform`) a validación y test.
2. **Valores faltantes:** la documentación indica que no hay. Confírmenlo; si encuentran alguno, decidan y justifiquen una estrategia de imputación.
3. **Variables categóricas:** en este dataset todas las variables son numéricas, así que no hay nada que codificar. Indíquenlo explícitamente en el notebook. (`Age` podría tratarse como categórica por tener pocos valores distintos, pero si lo hacen deben justificarlo y comparar ambos enfoques).
4. **Escalamiento de predictores:** apliquen `StandardScaler`, `MinMaxScaler` u otro, y justifiquen. *Recuerden: las redes neuronales son muy sensibles a la escala de las entradas.*
5. **Transformaciones y *feature engineering* (opcional, recomendado):**
   - `log1p(Age)`, para reducir la asimetría de la edad.
   - **Relación agua/cemento** (`Water / Cement`), que es el factor más conocido en la ingeniería del concreto.
   - Variables como el total de material cementante (`Cement + Slag + Fly Ash`) o indicadores binarios de presencia (p. ej. `FlyAsh > 0`).

   Comparen el desempeño con y sin estas variables.
6. **Escalamiento del target (opcional, recomendado):** escalar `y` suele estabilizar el entrenamiento. Si lo hacen, **recuerden invertir la transformación** antes de calcular métricas y de generar el archivo de submission.

Se recomienda encapsular el preprocesamiento en un `Pipeline`/`ColumnTransformer` de `scikit-learn` para evitar *leakage* y aplicarlo igual a test.

### Parte 3 — `Dataset` y `DataLoader`

Basándose en `MyDataset` de [02_asl.ipynb](02_asl.ipynb):

- Conviertan `X` e `y` a tensores `torch.float32` y envíenlos al `device` (`cuda` si está disponible).
- Asegúrense de que `y` tenga forma `(N, 1)`.
- Creen `train_loader` (`shuffle=True`) y `valid_loader` (`shuffle=False`). Experimenten con el `batch_size`.

### Parte 4 — Arquitectura de la red

Diseñen una MLP **profunda** (al menos **3 capas ocultas**) con:

- `nn.Linear` + función de activación (`nn.ReLU`, `nn.LeakyReLU`, `nn.GELU`, ...).
- Salida de **1 neurona sin activación**.
- Opcionalmente: `nn.Dropout`, `nn.BatchNorm1d` y/o *weight decay* para regularizar.

Impriman el modelo y el número de parámetros entrenables.

### Parte 5 — Entrenamiento y validación

Adapten las funciones `train()` y `validate()` de los notebooks:

- Función de pérdida de regresión y optimizador (`Adam` u otro; justifiquen el *learning rate*).
- Entrenen por un número razonable de épocas registrando **loss de entrenamiento y de validación** en cada época.
- Calculen en validación: **MSE** (métrica oficial), y como métricas complementarias **RMSE, MAE y R²**, todas en la escala original del target (MPa).
- Grafiquen las **curvas de aprendizaje** (train vs. validation loss) y discutan si hay *underfitting* u *overfitting*.
- Implementen **early stopping** (o guarden el mejor modelo según la pérdida de validación).

### Parte 6 — Experimentación

Realicen **al menos 3 experimentos** variando hiperparámetros o decisiones de diseño, por ejemplo:

- Número de capas y neuronas
- Función de activación
- Learning rate / optimizador / batch size
- Dropout / BatchNorm / weight decay
- Tipo de escalamiento o transformación de variables

Resuman los resultados en una **tabla comparativa** (configuración → MSE de validación). Como el dataset es pequeño, el MSE de una sola partición puede variar bastante según la semilla. Se recomienda reportar el **promedio ± desviación estándar con K-Fold (p. ej. K = 5)**.

Como referencia, comparen su mejor red contra un **baseline sencillo** (p. ej. predecir la media, o una `LinearRegression`). ¿La red lo supera?

### Parte 7 — Predicción y submission en Kaggle

1. Con la mejor configuración, (opcionalmente) reentrenen con todo `train.csv`.
2. Apliquen **exactamente el mismo preprocesamiento** a `test.csv`.
3. Generen predicciones en la escala original y construyan el archivo con el mismo formato de `sample_submission.csv`.
4. Suban el archivo a Kaggle y reporten el **puntaje público** obtenido (incluyan un pantallazo).

---

## 5. Entregables

1. **Notebook** (`.ipynb`) ejecutado de principio a fin, con celdas Markdown que expliquen cada decisión. Debe correr sin errores con "Restart & Run All".
2. Archivo **`submission.csv`** generado.
3. **Pantallazo** del puntaje en el leaderboard de Kaggle.
4. **Conclusiones** (al final del notebook, 1–2 párrafos): qué funcionó, qué no, y qué harían con más tiempo.

---

## 6. Rúbrica de evaluación

| Criterio | Peso |
|---|---|
| EDA de reconocimiento y hallazgos que justifican el preprocesamiento | 20 % |
| Preprocesamiento correcto (escalamiento, encoding si aplica, sin *data leakage*) | 20 % |
| Implementación desde cero en PyTorch (`Dataset`, `DataLoader`, modelo, `train`/`validate`) adaptada a regresión | 25 % |
| Experimentación, curvas de aprendizaje y análisis de resultados | 20 % |
| Submission en Kaggle, claridad del notebook y conclusiones | 15 % |

---

## 7. Recomendaciones y errores comunes

- **No escalen antes de partir los datos.** Ajustar el scaler con todo el dataset filtra información de validación.
- **Forma del target:** `MSELoss` con predicciones `(N, 1)` y target `(N,)` hace *broadcasting* silencioso y produce resultados incorrectos. Usen `y.view(-1, 1)`.
- **`model.train()` y `model.eval()`:** cámbienlos correctamente, sobre todo si usan Dropout o BatchNorm. En validación usen `torch.no_grad()`.
- **Inviertan el escalamiento del target** antes de reportar métricas o subir predicciones.
- Fijen semillas (`torch.manual_seed`, `np.random.seed`) para que sus resultados sean reproducibles.
- Empiecen con un modelo pequeño que funcione y luego aumenten la complejidad.
