import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de la página
st.set_page_config(
    page_title="📈 Forecasting Ventas 2025",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar el diseño
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
        height: 3rem;
    }
    .section-divider {
        border-top: 2px solid #667eea;
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Cargar datos de inferencia y modelo"""
    try:
        # Cargar dataframe de inferencia (ruta relativa desde app/)
        df = pd.read_csv('data/processed/inferencia_df_transformado.csv')
        df['fecha'] = pd.to_datetime(df['fecha'])
        return df
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        return None

@st.cache_resource
def load_model():
    """Cargar modelo entrenado"""
    try:
        model = joblib.load('models/modelo_final.joblib')
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {e}")
        return None

def aplicar_ajustes(df_producto, descuento_porcentaje, escenario_competencia):
    """Aplicar ajustes de descuento y competencia al dataframe"""
    df_ajustado = df_producto.copy()
    
    # Aplicar descuento al precio de venta
    df_ajustado['precio_venta'] = df_ajustado['precio_base'] * (1 + descuento_porcentaje/100)
    
    # Aplicar escenario de competencia
    factor_competencia = 1.0
    if escenario_competencia == "Competencia -5%":
        factor_competencia = 0.95
    elif escenario_competencia == "Competencia +5%":
        factor_competencia = 1.05
    
    # Ajustar precios de competencia
    for comp in ['Amazon', 'Decathlon', 'Deporvillage']:
        df_ajustado[comp] = df_ajustado[comp] * factor_competencia
    
    # Recalcular variables derivadas
    df_ajustado['precio_competencia'] = df_ajustado[['Amazon', 'Decathlon', 'Deporvillage']].mean(axis=1)
    df_ajustado['descuento_porcentaje'] = ((df_ajustado['precio_venta'] - df_ajustado['precio_base']) / df_ajustado['precio_base']) * 100
    df_ajustado['ratio_precio'] = df_ajustado['precio_venta'] / df_ajustado['precio_competencia']
    
    return df_ajustado

def predecir_recursivo(df_ajustado, model):
    """Realizar predicciones recursivas día por día"""
    df_pred = df_ajustado.copy()
    df_pred = df_pred.sort_values('fecha').reset_index(drop=True)
    
    predicciones = []
    
    # Variables de lag para tracking
    lag_cols = [f'unidades_vendidas_lag_{i}' for i in range(1, 8)]
    
    for idx in range(len(df_pred)):
        # Preparar datos para predicción (excluir columnas que no usa el modelo)
        excluir = ['fecha', 'producto_id', 'nombre', 'categoria', 'subcategoria', 'unidades_vendidas', 'ingresos']
        X_pred = df_pred.iloc[idx:idx+1].drop(columns=[col for col in excluir if col in df_pred.columns])
        
        # Asegurar que tenemos las columnas correctas para el modelo
        try:
            # Verificar si las columnas coinciden con las esperadas por el modelo
            model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_pred.columns
            X_pred = X_pred.reindex(columns=model_features, fill_value=0)
            
            # Hacer predicción
            pred = model.predict(X_pred)[0]
            pred = max(0, pred)  # Asegurar que no sea negativo
            predicciones.append(pred)
            
            # Actualizar lags para el siguiente día (si no es el último)
            if idx < len(df_pred) - 1:
                # Actualizar lags desplazando valores
                for lag_num in range(7, 1, -1):
                    if lag_num == 2:
                        df_pred.loc[idx + 1, f'unidades_vendidas_lag_{lag_num}'] = df_pred.loc[idx, 'unidades_vendidas_lag_1']
                    elif lag_num > 2:
                        df_pred.loc[idx + 1, f'unidades_vendidas_lag_{lag_num}'] = df_pred.loc[idx, f'unidades_vendidas_lag_{lag_num-1}']
                
                # El lag_1 del siguiente día es la predicción actual
                df_pred.loc[idx + 1, 'unidades_vendidas_lag_1'] = pred
                
                # Actualizar media móvil con las últimas predicciones
                if idx >= 6:  # Tenemos al menos 7 predicciones
                    ultimas_7 = predicciones[-7:]
                else:
                    # Combinar predicciones existentes con lags anteriores
                    predicciones_disponibles = predicciones.copy()
                    # Agregar lags del día actual para completar la ventana
                    for lag_num in range(len(predicciones_disponibles), 7):
                        if f'unidades_vendidas_lag_{lag_num + 1}' in df_pred.columns:
                            valor_lag = df_pred.loc[idx, f'unidades_vendidas_lag_{lag_num + 1}']
                            if pd.notna(valor_lag):
                                predicciones_disponibles.append(valor_lag)
                    
                    ultimas_7 = predicciones_disponibles[-7:]
                
                if len(ultimas_7) > 0:
                    df_pred.loc[idx + 1, 'unidades_vendidas_ma_7'] = np.mean(ultimas_7)
        
        except Exception as e:
            st.error(f"Error en predicción día {idx + 1}: {e}")
            predicciones.append(0)
    
    return predicciones

def crear_grafico_prediccion(df_pred, predicciones):
    """Crear gráfico de predicción con Black Friday destacado"""
    # Configurar el estilo de seaborn
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dias = range(1, len(predicciones) + 1)
    
    # Línea principal de predicción
    ax.plot(dias, predicciones, linewidth=3, color='#667eea', marker='o', markersize=4)
    
    # Destacar Black Friday (día 29 - último viernes de noviembre 2025)
    black_friday_dia = 28  # 28 de noviembre de 2025 es viernes
    if black_friday_dia <= len(predicciones):
        # Línea vertical
        ax.axvline(x=black_friday_dia, color='red', linestyle='--', alpha=0.7, linewidth=2)
        # Punto destacado
        ax.scatter(black_friday_dia, predicciones[black_friday_dia-1], color='red', s=100, zorder=5)
        # Anotación
        ax.annotate('🛍️ Black Friday', xy=(black_friday_dia, predicciones[black_friday_dia-1]), 
                   xytext=(black_friday_dia+2, predicciones[black_friday_dia-1]+1),
                   arrowprops=dict(arrowstyle='->', color='red'), fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Día de Noviembre 2025', fontsize=12, fontweight='bold')
    ax.set_ylabel('Unidades Vendidas Predichas', fontsize=12, fontweight='bold')
    ax.set_title('📊 Predicción Diaria de Ventas - Noviembre 2025', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Mejorar el aspecto
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    return fig

def main():
    # Header principal
    st.markdown("""
    <div class="main-header">
        <h1>📈 Simulador de Ventas Noviembre 2025</h1>
        <p>Predicciones inteligentes con Machine Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cargar datos y modelo
    df = load_data()
    model = load_model()
    
    if df is None or model is None:
        st.stop()
    
    # Sidebar con controles
    st.sidebar.markdown("## 🎛️ Controles de Simulación")
    
    # Selector de producto
    productos = sorted(df['nombre'].unique())
    producto_seleccionado = st.sidebar.selectbox(
        "📦 Seleccionar Producto",
        productos,
        index=0
    )
    
    # Slider de descuento
    descuento = st.sidebar.slider(
        "💰 Ajuste de Descuento (%)",
        min_value=-50,
        max_value=50,
        value=0,
        step=5,
        help="Ajusta el descuento sobre el precio base"
    )
    
    # Selector de escenario de competencia
    st.sidebar.markdown("🏪 **Escenario de Competencia**")
    escenario = st.sidebar.radio(
        "",
        ["Actual (0%)", "Competencia -5%", "Competencia +5%"],
        help="Simula cambios en los precios de la competencia"
    )
    
    # Botón de simulación
    st.sidebar.markdown("---")
    simular = st.sidebar.button("🚀 Simular Ventas", type="primary")
    
    if simular:
        with st.spinner('🔮 Generando predicciones recursivas...'):
            # Filtrar datos para el producto seleccionado
            df_producto = df[df['nombre'] == producto_seleccionado].copy()
            
            if df_producto.empty:
                st.error("No se encontraron datos para el producto seleccionado")
                return
            
            # Aplicar ajustes
            df_ajustado = aplicar_ajustes(df_producto, descuento, escenario)
            
            # Realizar predicciones recursivas
            predicciones = predecir_recursivo(df_ajustado, model)
            
            # Calcular métricas
            unidades_totales = sum(predicciones)
            precio_promedio = df_ajustado['precio_venta'].mean()
            ingresos_totales = sum(pred * precio for pred, precio in zip(predicciones, df_ajustado['precio_venta']))
            descuento_promedio = df_ajustado['descuento_porcentaje'].mean()
        
        # Header del dashboard
        st.markdown(f"""
        <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 2rem;">
            <h2 style="color: #667eea; margin: 0;">📊 Dashboard - {producto_seleccionado}</h2>
            <p style="margin: 0.5rem 0 0 0; color: #666;">Simulación para Noviembre 2025 • Escenario: {escenario} • Descuento: {descuento:+.0f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
        # KPIs principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "🎯 Unidades Totales",
                f"{unidades_totales:.0f}",
                delta=None
            )
        
        with col2:
            st.metric(
                "💰 Ingresos Proyectados",
                f"€{ingresos_totales:,.2f}",
                delta=None
            )
        
        with col3:
            st.metric(
                "🏷️ Precio Promedio",
                f"€{precio_promedio:.2f}",
                delta=None
            )
        
        with col4:
            st.metric(
                "📊 Descuento Promedio",
                f"{descuento_promedio:+.1f}%",
                delta=None
            )
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Gráfico de predicción
        st.markdown("### 📈 Predicción Diaria de Ventas")
        fig = crear_grafico_prediccion(df_ajustado, predicciones)
        st.pyplot(fig)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Tabla detallada
        st.markdown("### 📋 Detalle Diario de Predicciones")
        
        # Crear tabla de resultados
        tabla_resultados = []
        for idx, (_, row) in enumerate(df_ajustado.iterrows()):
            es_black_friday = row['es_BlackFriday']
            emoji_dia = "🛍️" if es_black_friday else ""
            
            tabla_resultados.append({
                "📅 Fecha": row['fecha'].strftime('%d/%m/%Y'),
                "📆 Día Semana": row['nombre_dia'],
                "💰 Precio Venta": f"€{row['precio_venta']:.2f}",
                "🏪 Precio Competencia": f"€{row['precio_competencia']:.2f}",
                "📊 Descuento": f"{row['descuento_porcentaje']:+.1f}%",
                "📦 Unidades": f"{predicciones[idx]:.0f}",
                "💵 Ingresos": f"€{predicciones[idx] * row['precio_venta']:.2f}",
                "🎉": emoji_dia
            })
        
        df_tabla = pd.DataFrame(tabla_resultados)
        st.dataframe(df_tabla, use_container_width=True, height=400)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Comparativa de escenarios
        st.markdown("### 🔄 Comparativa de Escenarios de Competencia")
        st.markdown("*Manteniendo el descuento actual, comparamos diferentes escenarios de precios de competencia*")
        
        col1, col2, col3 = st.columns(3)
        
        escenarios_comparar = ["Actual (0%)", "Competencia -5%", "Competencia +5%"]
        
        for idx, esc in enumerate(escenarios_comparar):
            with [col1, col2, col3][idx]:
                # Simular cada escenario
                df_temp = aplicar_ajustes(df_producto, descuento, esc)
                pred_temp = predecir_recursivo(df_temp, model)
                unidades_temp = sum(pred_temp)
                ingresos_temp = sum(pred * precio for pred, precio in zip(pred_temp, df_temp['precio_venta']))
                
                # Calcular diferencias
                diff_unidades = unidades_temp - unidades_totales if esc != escenario else 0
                diff_ingresos = ingresos_temp - ingresos_totales if esc != escenario else 0
                
                # Color según el escenario
                color = "#667eea" if esc == escenario else "#95a5a6"
                
                st.markdown(f"""
                <div style="background: {color}; color: white; padding: 1rem; border-radius: 8px; text-align: center;">
                    <h4 style="margin: 0; color: white;">{esc}</h4>
                    <p style="margin: 0.5rem 0; font-size: 1.2rem; font-weight: bold;">{unidades_temp:.0f} unidades</p>
                    <p style="margin: 0; font-size: 1rem;">€{ingresos_temp:,.0f}</p>
                    {f'<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">({diff_unidades:+.0f} unidades, €{diff_ingresos:+,.0f})</p>' if diff_unidades != 0 else '<p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">📍 Escenario actual</p>'}
                </div>
                """, unsafe_allow_html=True)
        
    else:
        # Pantalla de bienvenida
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 10px;">
            <h2>🎯 ¡Bienvenido al Simulador de Ventas!</h2>
            <p style="font-size: 1.1rem; color: #666; margin: 1rem 0;">
                Utiliza los controles del panel izquierdo para simular diferentes escenarios de ventas para noviembre 2025.
            </p>
            <p style="color: #667eea; font-weight: bold;">
                📦 Selecciona un producto<br>
                💰 Ajusta el descuento<br>  
                🏪 Elige un escenario de competencia<br>
                🚀 ¡Haz clic en "Simular Ventas"!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Información adicional
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🤖 IA Predictiva**
            - Modelo: HistGradientBoosting
            - Predicciones recursivas día a día
            - Actualización automática de lags
            """)
        
        with col2:
            st.markdown("""
            **📊 Variables Consideradas**
            - Tendencias históricas
            - Estacionalidad y festivos
            - Precios de competencia
            - Black Friday y eventos especiales
            """)
        
        with col3:
            st.markdown("""
            **🎯 Funcionalidades**
            - 24 productos disponibles
            - Ajustes de descuento (-50% a +50%)
            - 3 escenarios de competencia
            - Comparativas automatizadas
            """)

if __name__ == "__main__":
    main()
