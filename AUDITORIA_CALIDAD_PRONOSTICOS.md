# Auditoría de calidad de pronósticos — 12 de octubre de 2026

## Alcance y evidencia actual

Auditoría generada desde los CSV de la rama `main` después de ejecutar liquidadores y reconciliación en GitHub Actions. La reconciliación cubre **7.292 de 7.292 IDs de predicción** de los cuatro deportes incluidos (LaLiga, Champions, MLB y NFL). Esto valida cobertura del informe, no que cada pronóstico tenga ya un resultado terminal.

- Pipeline de liquidación y reconciliación: [ejecución 38221151345](https://github.com/fall2606/predicciones/actions/runs/38221151345), completada correctamente.
- Pruebas unitarias: [ejecución 38221151346](https://github.com/fall2606/predicciones/actions/runs/38221151346), completadas correctamente.
- Verificación del sistema: [ejecución 38221151428](https://github.com/fall2606/predicciones/actions/runs/38221151428), completada correctamente.
- La URL desplegada de Streamlit no pudo verificarse visualmente en esta auditoría. No se afirma que producción ya muestre estos cambios.

Las métricas son un diagnóstico descriptivo de capturas históricas, no un backtest cronológico fuera de muestra. No demuestran rentabilidad ni que la calibración haya aprendido sin fuga temporal.

## Calidad por deporte

| Deporte | Capturas | Resultados WIN/LOSS evaluables | Acierto | Probabilidad media | Brier | Referencia simple de la muestra | Evaluación |
|---|---:|---:|---:|---:|---:|---:|---|
| LaLiga | 532 | 293 | 56,3% | 72,4% | 0,256 | 0,246 | No supera la referencia simple |
| Champions League | 204 | 0 | N/D | N/D | N/D | N/D | Sin resultados evaluables; las 204 capturas aparecen como no finalizadas en la reconciliación actual |
| MLB | 3.741 | 24 | 29,2% | 74,6% | 0,391 | 0,207 | Muestra pequeña y error probabilístico alto |
| NFL | 2.815 | 132 | 50,0% | 50,0% | 0,121 | 0,250 | Señal descriptiva; no equivale a validación temporal |

**Interpretación conservadora:**

- **LaLiga:** el modelo asigna de media 72,4% de probabilidad, pero acierta 56,3% de los casos evaluados. Su Brier es peor que la referencia simple de la muestra. No se debe tratar la probabilidad alta como prueba de ventaja.
- **Champions:** aún no se puede medir la calidad; no hay filas terminales en `resultados_champions.csv`. Las 204 capturas están clasificadas como partidos no finalizados por la fuente consultada, así que no cuentan como pendientes vencidas en este corte.
- **MLB:** solo 24 resultados evaluables frente a 3.741 capturas; la muestra no alcanza para validar el modelo. Además, el informe clasifica 3.650 registros históricos sin partido asociado como no evaluables, no como resultados resueltos.
- **NFL:** hay 132 resultados evaluables y 1.440 capturas excluidas por ser capturas históricas no válidas para el modelo inicial o mercados de periodo no cubiertos. Aunque el Brier descriptivo es inferior a la referencia simple, no hay un backtest cronológico fuera de muestra que permita afirmar que el modelo está validado.

La referencia Brier mostrada aquí es la tasa de acierto de la muestra usada como referencia descriptiva; no sustituye una comparación con precios de mercado ni un backtest temporal.

## Reconciliación de resultados

El informe canónico tiene una fila de reconciliación por cada ID de predicción. Estado de la última ejecución:

| Deporte | Liquidadas | No finalizadas / dentro de ventana | Pendientes activas | Histórico no evaluable |
|---|---:|---:|---:|---:|
| LaLiga | 293 | 230 | 9 | 0 |
| Champions League | 0 | 204 | 0 | 0 |
| MLB | 24 | 66 | 1 | 3.650 |
| NFL | 132 | 1.243 | 0 | 1.440 |
| **Total** | **449** | **1.743** | **10** | **5.090** |

Las 230 filas de LaLiga incluyen 201 partidos no finalizados y 29 filas sin ticker todavía dentro de la ventana normal. Los registros marcados como históricos no evaluables no son WIN/LOSS/ANULADA: se conservan separados porque les faltan datos de partido suficientes o pertenecen a mercados que el modelo no cubre. No se presentan como resultados liquidados.

Las 10 alertas activas actuales son:
- 9 predicciones de LaLiga de partidos terminados sin ticker real y sin resultado terminal.
- 1 predicción MLB finalizada cuyo contrato de Kalshi aún no aparecía resuelto en la última reconciliación.

La reconciliación ya no clasifica un mercado de Kalshi desaparecido/cerrado como resuelto solo por estar ausente: si el partido terminó y no existe un resultado deportivo verificable, mantiene una alerta activa. La página tampoco muestra cero si no existe cobertura completa del deporte.

## Defectos corregidos en esta revisión

1. La auditoría y la interfaz ahora exigen cobertura completa de IDs antes de publicar un contador de pendientes. Si falta cobertura, muestran que el contador no es verificable en vez de inventar un cero.
2. La reconciliación ahora incluye NFL, coteja sus eventos con el marcador ESPN por equipos y fecha, deduplica eventos repetidos y conserva las capturas excluidas en una categoría no evaluable.
3. Un mercado de Kalshi ausente/cerrado ya no se considera resuelto automáticamente. Si el partido finalizó y no hay un resultado deportivo verificable, queda como pendiente activo con motivo.
4. Se añadieron pruebas de regresión para mercados desaparecidos, cobertura incompleta y emparejamiento NFL por equipos/fecha.
5. NFL calcula el precio de entrada usando el ask ejecutable; no sustituye el ask por el bid.
6. LaLiga y Champions recalculan el edge desde la probabilidad calibrada y el precio de compra, y no guardan una recomendación sencilla cuando falta cualquiera de esos datos o no alcanza el umbral configurado.

## Lo que todavía falta antes de declarar la app lista

1. Resolver las 10 alertas activas y verificar que cada una termina en WIN, LOSS o ANULADA cuando exista evidencia suficiente. No se debe forzar una etiqueta cuando la fuente no permite determinar el resultado.
2. Auditar los 5.090 registros históricos no evaluables y decidir cuáles son mercados de temporada/filas antiguas sin partido y cuáles se pueden reconstruir de forma fiable. No deben contarse como liquidados.
3. Construir un backtest cronológico fuera de muestra por deporte y familia de mercado, comparando Brier, log-loss, calibración, precio ask de Kalshi, tarifas, ROI neto y CLV.
4. Mantener fuera de las recomendaciones cualquier mercado sin ticker real, precio ejecutable, probabilidad válida y ventaja neta mínima. Si el modelo no tiene evidencia suficiente, la interfaz debe decir “sin recomendación” y mostrar el motivo.
5. Confirmar visualmente la app desplegada después de la actualización de Streamlit. Los tests de GitHub no prueban que producción haya cambiado.

## Conclusión

La integridad de la reconciliación y la cobertura del informe mejoraron y las pruebas de GitHub pasaron. **La app aún no puede declararse lista como sistema de recomendaciones:** la calidad predictiva no está validada fuera de muestra, hay 10 alertas activas y 5.090 registros históricos no evaluables que requieren revisión. El despliegue visual tampoco está confirmado.
