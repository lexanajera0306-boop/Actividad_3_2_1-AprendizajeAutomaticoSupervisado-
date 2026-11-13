# Clasificador de muestras de mama — Actividad 3.2.1

Aplicación académica de **aprendizaje automático supervisado** realizada con Python y Streamlit. Utiliza el dataset público *Breast Cancer Wisconsin (Diagnostic)*, compara KNN, SVM, Random Forest y Árbol de Decisión, y emplea un modelo SVM para clasificar diez mediciones numéricas.

## Interfaz

- Presentación sencilla, sin efectos o decoraciones que distraigan.
- Tres ejemplos de datos: benigno, maligno y general.
- Diez mediciones editables (ocultas inicialmente para simplificar el uso).
- Clasificación, salida matemática y explicación de sus limitaciones.
- Pestañas de método y resultados, con una gráfica que compara una métrica a la vez.

## Ejecutar en Windows

Descomprime el ZIP. En la carpeta que contiene `app.py`, escribe `cmd` en la barra de direcciones. Ejecuta:

```bat
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abre `http://localhost:8501`. Si tienes la aplicación anterior abierta, detenla primero con `Ctrl + C` en su CMD.

## Probar el modelo

```bat
python pruebas_modelo.py
```

La prueba espera matriz de confusión `[[71, 1], [2, 40]]` sobre 114 registros.

## Archivos principales

- `app.py`: aplicación Streamlit.
- `.streamlit/config.toml`: colores y tema.
- `models/modelo.joblib`: modelo SVM entrenado.
- `models/metadata.json`: variables, rangos y métricas.
- `data/`: dataset y resultados comparativos.
- `actividad_3_2_1.ipynb`: notebook de trabajo.
- `entrenar_modelos.py`: entrenamiento reproducible.
- `pruebas_modelo.py`: pruebas automáticas.

## Alcance y uso responsable

La herramienta **no realiza diagnósticos médicos**; trabaja con datos públicos de investigación y ejemplos estadísticos. No deben introducirse datos de salud de personas reales.

**Declaración de herramientas:** Como requiere la actividad, el equipo debe incluir en su informe la declaración real de las herramientas de IA generativa que utilizó para desarrollar, revisar o rediseñar el programa, así como las verificaciones realizadas por sus integrantes. La estética de la interfaz no modifica esa obligación.
