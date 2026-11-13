# Clasificador de tumores de mama — Actividad 3.2.1

Proyecto de **aprendizaje automático supervisado** (clasificación binaria) para la materia *Fundamentos de Inteligencia Artificial* (Ingeniería en Tecnologías de la Información e Innovación Digital, 7.º cuatrimestre).
A partir de 10 medidas de núcleos celulares (biopsia FNA), el sistema predice si un tumor es **benigno** o **maligno**.

> ⚠️ Proyecto educativo. No es un dispositivo médico ni sustituye un diagnóstico profesional.

## 🔗 Aplicación en línea
**https://clasificador-mama-fkecmebrlt8ucmh3azmbq7.streamlit.app**
(Si la app está "dormida", presiona el botón para reactivarla; tarda cerca de un minuto.)

## Equipo
- Bernal Nájera Alexandra
- Luis Antonio Román Rey
- Fernando Yair Sanvicente Valencia
- Melonie Suárez

## Entregables
| # | Entregable | Dónde está |
|---|---|---|
| 1 | Ficha del problema y dataset | Sección 1 de `docs/informe_metodologico.pdf` y este README |
| 2 | Análisis y preparación de datos | `actividad_3_2_1.ipynb`, carpeta `figures/` y sección 2 del informe |
| 3 | Notebook con tres o más modelos y tabla comparativa | `actividad_3_2_1.ipynb` y `data/tabla_comparativa.csv` |
| 4 | Aplicación funcional | `app.py`, `models/` y el enlace de arriba |
| 5 | Informe metodológico (PDF) y presentación ejecutiva | `docs/informe_metodologico.pdf` y `docs/presentacion_ejecutiva.pptx` |
| 6 | Repositorio con README | Este repositorio |

## Problema y datos
- **Dataset:** Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository (incluido en `scikit-learn`). https://doi.org/10.24432/C5DW2B
- **Registros:** 569 (357 benignos / 212 malignos) · **Variables:** 10 medidas "media" · **Objetivo:** `diagnosis` (1 = maligno).
- **Tipo de problema:** clasificación. **Métrica prioritaria:** Recall de la clase maligna.

## Resultados (conjunto de prueba, 114 casos)
| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| **SVM (ganador)** | 0.974 | 0.976 | 0.952 | 0.964 | 0.989 |
| KNN | 0.947 | 0.909 | 0.952 | 0.930 | 0.975 |
| Random Forest | 0.939 | 0.927 | 0.905 | 0.916 | 0.985 |
| Árbol de Decisión | 0.921 | 0.923 | 0.857 | 0.889 | 0.928 |

El ganador se eligió con validación cruzada repetida (5 × 3) sobre el conjunto de entrenamiento; SVM y Random Forest quedaron prácticamente empatados. Detalles y limitaciones en el informe.

## Estructura
```
├── README.md
├── app.py                     # Aplicación web (Streamlit)
├── requirements.txt
├── actividad_3_2_1.ipynb      # EDA, preparación, modelos, evaluación y guardado del modelo
├── data/                      # Dataset en CSV y tablas de resultados
├── figures/                   # Gráficas generadas por el notebook
├── models/                    # modelo.joblib y metadata.json
└── docs/                      # Informe (PDF y Word) y presentación (PowerPoint)
```

## Cómo ejecutarlo en tu computadora
```bash
pip install -r requirements.txt
streamlit run app.py
```
En Windows, si `pip` no se reconoce: `py -m pip install -r requirements.txt` y `py -m streamlit run app.py`.
Para reproducir el análisis, abre `actividad_3_2_1.ipynb` en Google Colab y elige *Entorno de ejecución → Ejecutar todo*.

## Reproducibilidad
Semilla fija (`random_state=42`), partición estratificada 80/20, escalado dentro de un `Pipeline` (sin fuga de datos) y búsqueda de hiperparámetros con validación cruzada.

## Declaración de uso de IA generativa
Se utilizó **Claude (Anthropic)** como apoyo para estructurar el proyecto, generar código base y redactar borradores de documentación. El equipo revisó, ejecutó y adaptó el contenido.
