# Datos del Proyecto

Esta carpeta contiene todos los datos utilizados en el proyecto de forecasting de ventas.

## Estructura

### `/raw`
Contiene los datos originales sin procesar:
- Archivos CSV, Excel, JSON con datos de ventas
- Datos históricos de transacciones
- Información de productos y categorías
- Datos de clientes y regiones

**Importante**: No modifiques los archivos en esta carpeta. Mantén siempre una copia de los datos originales.

### `/processed`
Contiene los datos que han sido limpiados y transformados:
- Datos con valores faltantes imputados
- Variables categóricas codificadas
- Features de ingeniería aplicadas
- Datasets preparados para modelado

### `/external`
Datos obtenidos de fuentes externas:
- APIs de datos económicos
- Información de días festivos
- Datos de competencia
- Variables macroeconómicas

## Convenciones de Nomenclatura

- `ventas_raw_YYYYMMDD.csv` - Datos de ventas originales
- `ventas_cleaned_YYYYMMDD.csv` - Datos de ventas procesados
- `features_engineered_YYYYMMDD.csv` - Datos con features de ingeniería
- `train_set_YYYYMMDD.csv` - Conjunto de entrenamiento
- `test_set_YYYYMMDD.csv` - Conjunto de prueba

## Formato de Datos

Los datos principales deben contener al menos las siguientes columnas:
- `fecha`: Fecha de la transacción
- `producto_id`: Identificador del producto
- `categoria`: Categoría del producto
- `ventas`: Cantidad vendida o monto de ventas
- `region`: Región geográfica
- `cliente_id`: Identificador del cliente (opcional)

## Procesamiento de Datos

Consulta los scripts en `src/data/` para ver cómo se procesan los datos desde raw hasta processed.