# Ejercicio de clase — Pipeline + KNN (Regresión)

## Contexto

Hasta ahora han visto procesamiento de datos, transformación de columnas (`ColumnTransformer`) y el algoritmo KNN. En este ejercicio van a aplicar todo eso sobre un problema de **regresión**: predecir el precio de una vivienda a partir de sus características físicas.

Antes de construir el modelo, van a mirar la variable objetivo y justificar explícitamente por qué este es un problema de regresión y no de clasificación, y por qué usarán `KNeighborsRegressor` en lugar de `KNeighborsClassifier`.

## Dataset

Van a trabajar con el dataset de precios de vivienda, disponible en:

```
data/house_price_regression_dataset.csv
```

Cada fila es una vivienda. Columnas disponibles:

| Columna | Descripción |
|---|---|
| `Square_Footage` | Área construida (pies cuadrados) |
| `Num_Bedrooms` | Número de habitaciones |
| `Num_Bathrooms` | Número de baños |
| `Year_Built` | Año de construcción |
| `Lot_Size` | Tamaño del lote (acres) |
| `Garage_Size` | Número de puestos de garaje |
| `Neighborhood_Quality` | Puntaje de calidad del vecindario (1 a 10) |
| `House_Price` | **Variable objetivo** — precio de la vivienda (USD) |

Todas las columnas de este dataset son numéricas. Para poder practicar un `ColumnTransformer` con dos tipos de transformación (como en un caso real con variables mixtas), van a tratar:

- **Numéricas continuas** (`StandardScaler`): `Square_Footage`, `Year_Built`, `Lot_Size`.
- **Discretas de baja cardinalidad, tratadas como categóricas** (`OneHotEncoder`): `Num_Bedrooms`, `Num_Bathrooms`, `Garage_Size`, `Neighborhood_Quality`.

No hay una columna identificadora que deba excluirse, pero sí deben excluir `House_Price` de las *features* (es el target).

## Objetivos de aprendizaje

- Explorar un dataset y distinguir columnas numéricas continuas de columnas discretas/categóricas.
- Justificar, a partir de la variable objetivo, por qué un problema es de **regresión**.
- Construir un `Pipeline` con `ColumnTransformer` que escale las columnas continuas y codifique (`OneHotEncoder`) las discretas/categóricas.
- Entrenar un `KNeighborsRegressor` dentro de ese pipeline.
- Variar el parámetro de distancia `p` de Minkowski (`p=1` Manhattan, `p=2` Euclidiana, `p=3`) y comparar resultados.

## Instrucciones

### 1. Carga y exploración

- Cargar el CSV y revisar tipos de datos, nulos y estadísticas descriptivas (`describe()`).
- Confirmar que no hay valores nulos (si los hubiera, decidir cómo tratarlos).
- Listar explícitamente qué columnas se van a tratar como numéricas continuas y cuáles como discretas/categóricas.

### 2. ¿Por qué es un problema de regresión?

- Miren la columna `House_Price`: ¿qué tipo de valores toma?, ¿cuántos valores distintos tiene?
- En una celda de markdown, respondan: **¿por qué este problema es de regresión y no de clasificación?** y confirmen que usarán `KNeighborsRegressor`.

### 3. Partición de datos

- Separar `X` (features, sin `House_Price`) e `y` (`House_Price`).
- `train_test_split` con 80/20 y `random_state=42` fijo.

### 4. Pipeline de preprocesamiento + modelo

- Construir un `ColumnTransformer` que aplique `StandardScaler` a las columnas continuas y `OneHotEncoder(handle_unknown="ignore")` a las discretas/categóricas.
- Meter ese `ColumnTransformer` y el `KNeighborsRegressor` dentro de un único `Pipeline`.
- El modelo **siempre** se entrena a través del pipeline, nunca sobre datos transformados manualmente por fuera.

### 5. Variar el número de vecinos y el tipo de distancia

Todavía no hemos visto `GridSearchCV`, así que esta búsqueda se hace con un **`for` anidado**:

- Recorrer las combinaciones de:
  - `n_neighbors`: `[3, 5, 7, 9, 11, 15]`
  - `p` (parámetro de distancia Minkowski): `[1, 2, 3]` (`p=1` Manhattan, `p=2` Euclidiana, `p=3`)
- Para cada combinación, construir el pipeline con esos parámetros y calcular su desempeño de validación cruzada con `cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")`, tomando el promedio.
- Guardar los resultados (`n_neighbors`, `p`, `mean_r2` de cada combinación) en una tabla (`cv_results`) y armar un gráfico simple: R² promedio de CV vs. `n_neighbors`, una curva por cada valor de `p`.
- A partir de esa tabla, identificar la mejor combinación (`best_params` y `best_score`).

### 6. Evaluación

Sobre el conjunto de **prueba**, evaluar el mejor modelo encontrado con `MAE`, `RMSE` y `R²`. (Busca como calcularlos usando sklearn.metrics)



## Entregable

Un notebook (`.ipynb`) con:

1. Código y outputs de los pasos 1–6.
2. La celda de markdown del paso 2 justificando por qué es un problema de regresión.
3. La tabla y el gráfico comparando los valores de `p`.
4. Una conclusión corta (mínimo 80 palabras): **¿qué valor de `p` funcionó mejor y por qué creen que pasó eso con este dataset?**

> 💡 En [Ejercicio_KNN_Pipeline.ipynb](Ejercicio_KNN_Pipeline.ipynb) encuentran una guía paso a paso con celdas de código de partida (`# TODO`) para completar. Hay 3 **checkpoints** con `assert` a lo largo del notebook (tras el pipeline, tras la búsqueda con `for` y al final) para verificar que su implementación va por buen camino.

## Criterios de evaluación

| Criterio | Peso |
|---|---|
| Identificación correcta del problema como regresión y uso de `KNeighborsRegressor` | 20% |
| Pipeline de preprocesamiento correcto (numéricas + categóricas, sin fuga de datos) | 30% |
| Búsqueda correcta de `n_neighbors` y `p` con `for` + `cross_val_score` | 25% |
| Tabla/gráfico comparando distancias y métricas de evaluación | 15% |
| Claridad de la conclusión | 10% |
