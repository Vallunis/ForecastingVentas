# Forecasting Ventas - Proyecto de Machine Learning

## Descripción del Proyecto
Este proyecto implementa un sistema de forecasting de ventas utilizando técnicas de Machine Learning y análisis de series temporales. Incluye análisis exploratorio de datos, desarrollo de modelos predictivos y una aplicación web interactiva desarrollada con Streamlit.

## Estructura del Proyecto

```
ForecastingVentas/
│
├── data/                   # Datos del proyecto
│   ├── raw/               # Datos originales sin procesar
│   ├── processed/         # Datos procesados y limpios
│   └── external/          # Datos externos
│
├── notebooks/             # Jupyter notebooks
│   ├── exploratory/       # Análisis exploratorio de datos
│   └── modeling/          # Desarrollo y evaluación de modelos
│
├── src/                   # Código fuente
│   ├── data/              # Scripts para procesamiento de datos
│   ├── features/          # Scripts para ingeniería de características
│   ├── models/            # Scripts para entrenamiento de modelos
│   └── utils/             # Utilidades y funciones auxiliares
│
├── models/                # Modelos entrenados
│   ├── trained/           # Modelos entrenados (.pkl, .joblib)
│   └── exports/           # Modelos exportados para producción
│
├── app/                   # Aplicación Streamlit
│   ├── pages/             # Páginas de la aplicación
│   ├── components/        # Componentes reutilizables
│   └── utils/             # Utilidades para la app
│
├── config/                # Archivos de configuración
│
├── docs/                  # Documentación del proyecto
│
├── requirements.txt       # Dependencias del proyecto
├── .gitignore            # Archivos a ignorar por Git
└── README.md             # Documentación principal
```

## Instalación

1. Clona el repositorio:
```bash
git clone <repository-url>
cd ForecastingVentas
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Análisis de Datos
Los notebooks de análisis exploratorio se encuentran en `notebooks/exploratory/`

### Entrenamiento de Modelos
Los notebooks para desarrollo de modelos están en `notebooks/modeling/`

### Aplicación Web
Para ejecutar la aplicación Streamlit:
```bash
streamlit run app/main.py
```

## Estructura de Datos
- **raw/**: Datos originales en formato CSV, Excel o JSON
- **processed/**: Datos limpios y transformados listos para modelado
- **external/**: Datos de fuentes externas (APIs, bases de datos, etc.)

## Modelos Implementados
- Modelos estadísticos (ARIMA, SARIMA)
- Prophet para series temporales
- Modelos de Machine Learning (XGBoost, LightGBM)
- Ensemble methods

## Contribución
1. Fork del repositorio
2. Crear rama para nueva feature
3. Commit de cambios
4. Push a la rama
5. Crear Pull Request

## Licencia
[Especificar licencia]

## Contacto
[Información de contacto]