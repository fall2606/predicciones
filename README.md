# predicciones

## Fuentes de resultados y liquidación

- Kalshi sigue siendo la fuente de mercados, precios y probabilidades al generar las predicciones.
- La liquidación intenta primero usar football-data.org con el ID oficial del partido y su marcador reglamentario; la consulta por temporada se almacena en caché para no hacer una petición por predicción.
- Si football-data.org no está disponible o no identifica el partido, se usa como respaldo el marcador público de ESPN, con coincidencia conservadora de ambos equipos. El endpoint de ESPN es público pero no documentado y puede cambiar sin aviso.
- Los mercados que requieren datos no incluidos de forma fiable en el marcador final (corners, primera/segunda parte, primer gol, tarjetas y props) siguen usando la resolución de Kalshi. Si no hay coincidencia segura, no se inventa un resultado.
- Los partidos de Champions que el proveedor identifica como prórroga o penales se dejan a la resolución de Kalshi, porque el marcador reglamentario puede diferir del marcador final.
- SofaScore no se usa como fuente automatizada: sus condiciones restringen la extracción y reutilización de su base de datos sin permiso explícito.
- El workflow central `Pipeline automático de predicciones` comprueba la liquidación de MLB, LaLiga y Champions cada 5 minutos; los workflows individuales quedan disponibles para ejecución manual y no se programan en paralelo para evitar colas de concurrencia. Las ejecuciones programadas pueden retrasarse. Un resultado externo solo se guarda cuando el partido aparece finalizado y la regla del mercado se puede calcular sin ambigüedad.

## Historial de LaLiga

- El workflow **Historial inmutable de LaLiga** se ejecuta cada dos horas y también puede iniciarse manualmente desde **GitHub → Actions**.
- `predicciones_laliga.csv` guarda cada captura antes del comienzo. El proceso sólo añade filas; no cambia capturas anteriores.
- `resultados_laliga.csv` guarda los resultados en un archivo separado. La página compara la última captura previa al inicio y permite abrir todas las capturas del partido.
- La calibración empieza cuando una categoría tiene al menos 30 pronósticos comparables ya resueltos. Antes de ese mínimo, muestra la probabilidad base.
- Los partidos que terminaron antes de activar este archivo no tienen una captura histórica recuperable; la app los identifica en vez de inventarla.

## Champions League

- La página **Champions League** reutiliza el resumen de mercados Kalshi, márgenes, capturas inmutables y evaluación de resultados con su propio modelo basado en la competición CL.
- El workflow **Historial inmutable de Champions League** archiva pronósticos previos cada dos horas y puede iniciarse manualmente en GitHub Actions.
- Los resultados de los contratos de partido usan el marcador reglamentario de 90 minutos; prórroga y penales quedan fuera, de acuerdo con la resolución de los mercados de fútbol de Kalshi.
- El archivo de Champions se mantiene separado del de LaLiga. Por ahora football-data.org no aporta aquí corners históricos de Champions, así que se muestran contratos sin inventar sus probabilidades.


## NFL

- La página **NFL** utiliza un motor específico de fútbol americano, con proyecciones de puntos a partir de los resultados recientes de ESPN y mercados reales de Kalshi. Si no encuentra una muestra suficiente de ambos equipos, no inventa la proyección.
- `nfl_engine.py` captura antes del inicio y conserva las capturas originales en `predicciones_nfl.csv`; las probabilidades recalibradas se guardan separadas de la probabilidad original.
- `liquidar_nfl.py` consulta la resolución oficial de Kalshi y escribe WIN/LOSS/ANULADA en `resultados_nfl.csv`. Los contratos sin resolver permanecen pendientes y no entran en calibración.
- El workflow **NFL - capturar, liquidar y recalibrar** se ejecuta cada 15 minutos y también admite ejecución manual. Los bloques de 5 sencillas, 5 para combos y 10 a tener en cuenta solo muestran selecciones que superan los filtros del modelo; si no hay suficientes, la página lo indica.
- La calibración se separa por lado YES/NO y categoría, y exige al menos 30 casos resueltos comparables antes de ajustar probabilidades.


## Contexto deportivo gratuito

- La página **Contexto deportivo** (pages/Contexto_deportivo.py) consulta Google News RSS para buscar noticias públicas por equipo o jugador, sin API key ni servicio de IA de pago.
- Presenta titulares, enlaces, fecha publicada y posibles temas a revisar (lesiones, alineaciones, sanciones, forma, clima o cambios), con caché de 15 minutos.
- Las etiquetas son reglas de palabras clave, no una IA que verifique hechos. Hay que abrir la fuente y confirmar que la noticia es actual y aplica al partido.
- Este módulo es deliberadamente independiente: **no modifica probabilidades, picks ni calibración WIN/LOSS**. No se aplican ajustes automáticos sin validación retrospectiva, para evitar contaminar el modelo con noticias falsas, duplicadas o irrelevantes.
- El RSS público puede limitarse o fallar; que no aparezcan noticias no significa que no haya bajas. Esta página no sustituye una API deportiva oficial de lesiones/alineaciones.

- El módulo `sports_context_reasoner.py` analiza antigüedad (72 horas), coincidencia de nombres, tipos de señal y posible ambigüedad. Es un motor determinista y auditable, no un LLM; nunca modifica probabilidades automáticamente. Tiene pruebas unitarias específicas.
