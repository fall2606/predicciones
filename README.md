# predicciones

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
