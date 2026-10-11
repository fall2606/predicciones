## NFL: alcance inicial del modelo y catálogo

La implementación NFL usa la API pública de Kalshi como autoridad para series, eventos, tickers, líneas, precios y reglas. No genera tickers sintéticos.

Modelo propio inicial: Elo cronológico con ventaja local y distribución de margen/total de puntos con incertidumbre NFL y tratamiento explícito de números clave. Las probabilidades pasan validación finita y [0,1]. La calibración por mercado solo se activará al alcanzar una muestra histórica mínima; sin datos suficientes se etiqueta «calibración insuficiente».

**Mercados elegibles para modelado inicial:** ganador del partido, spread de partido completo y total de puntos del partido completo, únicamente cuando se puede leer una línea/precio exactos de Kalshi y asociar sin ambigüedad al partido.

**Mercados que se catalogan pero no se pronostican todavía:** primera mitad/cuarto, totales por equipo y props de jugador. No se habilitan hasta que haya datos históricos/oficiales de período o jugador, reglas de liquidación verificadas y backtest por tipo de mercado. En particular, las bajas/inactivos y DNP no se estiman sin una fuente gratuita y verificable.

La prueba de catálogo es una comprobación en vivo: su salida debe mostrar los tickers y títulos devueltos realmente por Kalshi. El ejemplo no se fija en el código para evitar que quede obsoleto.
