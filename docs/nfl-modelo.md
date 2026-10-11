# NFL: catálogo Kalshi, modelo, filtros y reglas de liquidación

## Catálogo: Kalshi es la autoridad

El motor descubre series NFL mediante la API pública de Kalshi y obtiene eventos y mercados anidados. Antes de modelar un partido agrupa los contratos por fecha ET y los dos equipos; conserva ticker, línea, precios, estado y reglas primarias/secundarias del contrato. La página y el informe de catálogo separan:
- contratos calculables con precio;
- contratos calculables sin precio, etiquetados **«Sin precio en Kalshi»** y no elegibles para los bloques de valor;
- contratos no generados, con motivo explícito.

No se fabrican tickers, líneas, precios ni probabilidades. Los títulos y las reglas originales de Kalshi quedan guardados en el catálogo para auditar cómo define el umbral exacto de cada contrato.

## Modelo NFL propio

- Elo secuencial calculado cronológicamente, sin utilizar partidos posteriores al momento de captura.
- Ventaja local y proyección de puntos a favor/en contra con ponderación de forma reciente.
- Distribuciones discretas para margen y total de puntos, con números clave del fútbol americano (por ejemplo, 3 y 7).
- Diferencia de descanso entre equipos: el coeficiente se estima con residuos de márgenes históricos; si la muestra o la variación no es suficiente, se utiliza ajuste neutro.
- Validación de finitud y rango [0, 1] en todas las probabilidades. La escritura compartida conserva lado (YES/NO), probabilidad original, probabilidad calibrada, versión de calibración, precio, ventaja, bloque, captura y kickoff.

## Mercados que se generan y mercados que se excluyen

**Elegibles cuando la línea y el precio real están presentes y el emparejamiento no es ambiguo:** ganador del partido, spread de partido completo, total de puntos del partido y total por equipo. La ventaja se calcula respecto al precio comprador disponible, no respecto al bid.

**Catalogados, pero no pronosticados:** primeras/segundas mitades, cuartos y props de jugador (yardas de pase/carrera/recepción, recepciones y touchdowns). Se excluyen porque el modelo de producción no calcula todavía esos resultados con estadísticas oficiales por período/jugador y participación verificable. No se generan apuestas «no evaluables» ni se adivina el DNP. Si no se encuentra fuente fiable de inactivos, se declara esta limitación en lugar de inferir que el jugador participa.

## Resolución y reconciliación

1. **Primero Kalshi:** se consulta el resultado del ticker y su estado de liquidación. La página guarda las reglas primarias/secundarias de cada mercado para poder auditar el contrato.
2. **Respaldo ESPN:** solo para ganador, spread y totales de partido completo/equipo cuando se puede emparejar un único partido por equipos, fecha y kickoff. El marcador final oficial incluye la prórroga cuando ESPN la incluye.
3. **Mercados por período:** no se liquidan con el marcador final; por eso no se generan hasta tener una fuente verificable de puntos del período.
4. **Cancelado/suspendido o resultado no publicado:** se espera la resolución oficial de Kalshi o una cancelación oficial inequívoca. Si la coincidencia es ambigua o faltan datos, no se inventa WIN/LOSS. La reconciliación marca la fila y hace fallar el workflow si supera el plazo configurado.
5. **Jugador no participante (DNP):** no se generan props; la regla de anulación específica de Kalshi debe verificarse antes de habilitar ese tipo de contrato.

## Filtros y calibración

- Sencillas en tarjetas; tablas independientes «5 de alta probabilidad para combos» y «10 a tener en cuenta».
- Orden por ventaja, ventaja mínima configurable (5 puntos), tope de compra de 88¢ en Sencillas y 90¢ en alta probabilidad, diversificación por familia y protección contra correlaciones.
- No hay cupos fijos por tipo de mercado. Cada selección guardada en cada bloque tiene prediction_id distinto.
- Una ventaja superior a 20 puntos porcentuales queda marcada **REVISAR**; no se resta ventaja manualmente.
- El backtest walk-forward calcula Brier score y log loss por tipo y compara con un baseline 50/50. La calibración Platt exige muestra mínima (100 por mercado en la configuración actual); si no se alcanza, se muestra **«calibración insuficiente»**.
- **Limitación de datos:** el endpoint público consultado no conserva snapshots prepartido fiables de los precios de contratos NFL ya liquidados. Por eso el backtest histórico usa líneas proxy claramente etiquetadas y no afirma que sean precios históricos de Kalshi. La comparación con el precio real de Kalshi se activa cuando se disponga de snapshots prepartido guardados por el propio proyecto.

## Integridad del historial

Las capturas son append-only y se escriben en archivo temporal con reemplazo atómico. El escritor compartido rechaza capturas posteriores al kickoff y duplicados; el lector tolerante aparta filas mal formadas en un CSV de cuarentena. El histórico se conserva en una sección cerrada por defecto y no se borra al limitar el número de tarjetas visibles.
