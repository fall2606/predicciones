# Auditoría de calidad de pronósticos — 12 de octubre de 2026

## Alcance y límites

Revisión del código y los CSV disponibles en la rama `main` del repositorio. Las métricas se calcularon uniendo por `prediction_id` las capturas de LaLiga con sus resultados WIN/LOSS y usando `probabilidad_calibrada`. No se pudo comprobar visualmente la versión desplegada de Streamlit desde esta auditoría. Las cifras son un diagnóstico inicial, no un backtest cronológico completo.

## Hallazgos con evidencia

### 1. LaLiga: el porcentaje medio parece demasiado optimista

- Capturas archivadas: **532**.
- Capturas con resultado WIN/LOSS unido por ID: **293**.
- Filas aún sin resultado asociado: **239**. No todas son necesariamente atrasadas: antes de contarlas como pendientes vencidas hay que separar partidos futuros, activos, aplazados y terminados.
- Probabilidad media archivada: **72,4%**.
- Tasa de acierto de las 293 filas: **56,3%**.
- Brier calculado: **0,256** (menor es mejor).
- Referencia simple que asigna a cada fila la tasa de acierto de la muestra: Brier aproximado **0,246**.

La referencia no sustituye un backtest fuera de muestra, pero el modelo no la supera en esta revisión. La interfaz no debe dar a entender que una probabilidad alta ya significa fiabilidad demostrada.

### 2. Champions: no hay evaluación medible aún

- Capturas en `predicciones_champions.csv`: **203**.
- Filas en `resultados_champions.csv`: **0**.

Sin liquidaciones asociadas no se puede calcular la tasa de acierto ni el Brier de Champions ni afirmar que la calibración está aprendiendo de WIN/LOSS.

### 3. MLB: muestra insuficiente y señales de rendimiento débil

El archivo `backtest_mlb_resumen.csv` contiene muestras muy pequeñas. En los datos publicados allí:

| Mercado | Resultados | Brier modelo | Brier precio Kalshi |
|---|---:|---:|---:|
| Run line | 4 | 0,803 | 0,031 |
| Total de carreras | 15 | 0,301 | 0,096 |
| Primera entrada | 4 | 0,348 | 0,197 |

Son muestras demasiado pequeñas para una conclusión estable, pero no justifican recomendar estos mercados como si el modelo ya estuviera validado. Se necesita un backtest por fecha, mercado y selección que compare modelo, referencia simple y precio de mercado usando solo información disponible antes del partido.

### 4. Primer gol: error de modelado corregido en esta rama

En `laliga_engine.py` y `champions_engine.py`, la probabilidad de qué equipo anotaría primero se aproximaba usando la fracción de goles finales de cada equipo. El marcador final no contiene la secuencia temporal de los goles; esa fracción no es una probabilidad válida de primer anotador. La rama ahora deja ese mercado sin probabilidad propia, salvo el evento “no habrá goles”, que sí se puede estimar con la probabilidad de 0–0. Se añadieron pruebas para impedir que vuelva el cálculo engañoso.

### 5. Contexto no entra en las probabilidades

El módulo de noticias/contexto es independiente: muestra titulares y señales, pero no ajusta probabilidades. Los motores estadísticos no verifican automáticamente alineaciones, lesiones y clima antes de emitir cada pronóstico. La interfaz debe decirlo claramente, y el modelo no debe fingir que analizó esos factores.

### 6. Reconciliación y cobertura

El archivo `reconciliacion_pendientes.csv` tiene **4.476 filas**: 317 clasificadas como liquidadas, 10 pendientes activas, 499 no finalizadas y 3.650 históricos sin partido asociado. Hay **10 alertas** explícitas: 9 filas de LaLiga terminadas sin ticker y 1 fila MLB finalizada cuya resolución de Kalshi todavía no aparecía al generar el archivo. Las 3.650 filas sin ID de partido requieren deduplicación/clasificación; no deben presentarse como partidos actuales pendientes.

### 7. Interfaz

- La página de inicio anterior se titulaba Premier League aunque la app contiene varios deportes.
- Las páginas de fútbol presentaban varias tablas con métricas parecidas, difíciles de distinguir de los pronósticos.
- La página principal ahora funciona como selector de deportes; Premier League tiene página propia.
- Las listas principales de picks de LaLiga, Champions y NFL se están mostrando como tarjetas con probabilidad, precio, ventaja, resultado y ticker. Los datos de diagnóstico e historiales completos siguen disponibles en secciones secundarias.

## Criterios que deben cumplirse antes de llamar “bueno” a un modelo

1. Backtest cronológico fuera de muestra, por deporte y familia de mercado; nada de usar información posterior al inicio.
2. Reportar Brier, log-loss, tasa de acierto, tamaño de muestra y calibración por intervalos de probabilidad.
3. Comparar contra precio de Kalshi sin confundir bid con precio de compra ask, e incluir tarifas.
4. No recomendar contratos sin ticker real, precio de compra vigente, probabilidad válida y ventaja neta mínima.
5. Si el mercado o el modelo no tiene evidencia suficiente, mostrar “sin recomendación” con el motivo.
6. Para cada partido terminado, conciliar por una única regla de estado y registrar WIN/LOSS/ANULADA o un motivo específico verificable; no ocultar filas sin resolver.

## Estado de esta revisión

Esta auditoría detecta defectos y propone controles; **no demuestra que todos los pronósticos ya sean buenos ni que el despliegue esté verificado**. Las métricas deben regenerarse automáticamente desde los archivos de ejecución actuales, y las páginas desplegadas necesitan una comprobación visual después de que Streamlit termine de actualizar.
