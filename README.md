# predicciones

## Historial de LaLiga

- El workflow **Historial inmutable de LaLiga** se ejecuta cada dos horas y también puede iniciarse manualmente desde **GitHub → Actions**.
- `predicciones_laliga.csv` guarda cada captura antes del comienzo. El proceso sólo añade filas; no cambia capturas anteriores.
- `resultados_laliga.csv` guarda los resultados en un archivo separado. La página compara la última captura previa al inicio y permite abrir todas las capturas del partido.
- La calibración empieza cuando una categoría tiene al menos 30 pronósticos comparables ya resueltos. Antes de ese mínimo, muestra la probabilidad base.
- Los partidos que terminaron antes de activar este archivo no tienen una captura histórica recuperable; la app los identifica en vez de inventarla.
