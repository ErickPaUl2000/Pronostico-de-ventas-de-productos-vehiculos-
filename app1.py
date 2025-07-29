import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import Figure, Scatter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from utils_modelos import predecir_ventas, predecir_con_ets  
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# Cargar datos
def cargar_datos():
    df = pd.read_csv("SRI_Vehiculos_Nuevos_2024.csv", encoding='latin1', sep=';')
    df.columns = df.columns.str.strip().str.upper()
    col_fecha = next((col for col in df.columns if 'FECHA' in col and 'COMPRA' in col), None)
    df['FECHA COMPRA'] = pd.to_datetime(df[col_fecha], dayfirst=True, errors='coerce')
    if 'AVALÚO' in df.columns:
        df['AVALÚO'] = pd.to_numeric(df['AVALÚO'].astype(str).str.replace(',', ''), errors='coerce')
    df = df.drop_duplicates()
    df = df.dropna(subset=['FECHA COMPRA', 'AVALÚO'])
    df['MES_AÑO'] = df['FECHA COMPRA'].dt.to_period("M").astype(str)
    df['AÑO'] = df['FECHA COMPRA'].dt.year
    return df

df = cargar_datos()

# Configuración general
st.set_page_config(page_title="Pronóstico de Ventas de Autos", layout="wide")
st.markdown(
    """
    <style>
    .header-img {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 400px; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<img class="header-img" src="https://webct.internacional.edu.ec/MeritosOposicion/Images/nuevo_logo.png" alt="Logo">',
    unsafe_allow_html=True
)
st.title("📊 Análisis de Ventas de Vehículos Nuevos en Ecuador desde 2022 - 2024")
st.subheader("🔍 Filtros de Selección")
# Crear dos columnas
col1, col2 = st.columns([1, 1])

# Filtrado de datos
with col1:
    st.write("🔄 Marcas Disponibles")
    st.write("Selecciona las marcas para filtrar los datos.")
    marcas = st.multiselect("Selecciona marcas", sorted(df['MARCA'].unique()), default=list(df['MARCA'].unique()))
with col2:
    st.write("🔄 Meses Disponibles")
    st.write("Selecciona los meses para filtrar los datos.")
    # Filtrar meses disponibles desde 2022
    df['MES_AÑO'] = pd.to_datetime(df['MES_AÑO'], format='%Y-%m')
    available_months = sorted([month.strftime('%Y-%m') for month in df['MES_AÑO'].unique() if month.year >= 2022])
    meses = st.multiselect("Selecciona meses", available_months, default=available_months)

if st.button("🔄 Actualizar Resultados"):
    df_filtrado = df[(df['MARCA'].isin(marcas)) & (df['MES_AÑO'].isin(meses))].copy()

    # Resumen
    st.subheader("🔹 Resumen General")
    col1, col2, col3 = st.columns(3)
    col1.metric("Ventas Totales (Avalúo)", f"{df_filtrado['AVALÚO'].sum():,.0f}")
    col2.metric("Total Vehículos", f"{len(df_filtrado):,}")
    col3.metric("Precio Promedio", f"{df_filtrado['AVALÚO'].mean():,.2f}")

    # Gráfico de tendencia
    df_trend = df_filtrado.groupby(['MES_AÑO', 'MARCA'])['AVALÚO'].sum().reset_index()
    fig_trend = px.line(df_trend, x='MES_AÑO', y='AVALÚO', color='MARCA', title='Tendencia de Ventas por Mes y Marca')
    st.plotly_chart(fig_trend, use_container_width=True)

    # Distribuciones
    if 'TIPO COMBUSTIBLE' in df_filtrado.columns:
        combustible_counts = df_filtrado['TIPO COMBUSTIBLE'].value_counts().reset_index()
        combustible_counts.columns = ['TIPO_COMBUSTIBLE', 'CANTIDAD']
        fig_fuel = px.bar(
            combustible_counts,
            x='TIPO_COMBUSTIBLE',
            y='CANTIDAD',
            title='Distribución por Tipo de Combustible',
            labels={'TIPO_COMBUSTIBLE': 'Tipo de Combustible', 'CANTIDAD': 'Cantidad'}
    )
    st.plotly_chart(fig_fuel)


    if 'CLASE' in df_filtrado.columns:
        clase_counts = df_filtrado['CLASE'].value_counts().reset_index()
        clase_counts.columns = ['CLASE', 'CANTIDAD']

        # Gráfico
        fig_clase = px.bar(
        clase_counts,
        x='CLASE',
        y='CANTIDAD',
        title='Distribución por Clase de Vehículo',
        labels={'CLASE': 'Clase', 'CANTIDAD': 'Cantidad'}
    )
    st.plotly_chart(fig_clase)

    cluster_df = df_filtrado[['AVALÚO', 'AÑO']].dropna()
    if len(cluster_df) >= 3:
        scaled = StandardScaler().fit_transform(cluster_df)
        kmeans = KMeans(n_clusters=3, random_state=52, n_init=10)
        cluster_df['CLUSTER'] = kmeans.fit_predict(scaled)
        medios = cluster_df.groupby('CLUSTER')['AVALÚO'].mean().sort_values()
        etiquetas = {i: l for i, l in zip(medios.index, ['Económico', 'Intermedio', 'Lujo'])}
        cluster_df['CATEGORÍA'] = cluster_df['CLUSTER'].map(etiquetas)
        fig_cluster = px.scatter(cluster_df, x='AVALÚO', y='AÑO', color='CATEGORÍA', title='Clustering por Avalúo y Año')
        st.plotly_chart(fig_cluster, use_container_width=True)

    # Distribuciones adicionales
    if 'PAÍS' in df_filtrado.columns:
        pais_counts = df_filtrado['PAÍS'].value_counts().reset_index()
        pais_counts.columns = ['PAÍS', 'CANTIDAD']
        fig_country = px.bar(pais_counts,
                             x='PAÍS', y='CANTIDAD',
                             labels={'PAÍS': 'País', 'CANTIDAD': 'Cantidad'},
                             title='Distribución por País de Origen')
        st.plotly_chart(fig_country, use_container_width=True)

        avg_avaluo_country = df_filtrado.groupby('PAÍS')['AVALÚO'].mean().reset_index()
        fig_avg_country = px.bar(avg_avaluo_country, x='PAÍS', y='AVALÚO',
                                 title='Avalúo Promedio por País de Origen')
        st.plotly_chart(fig_avg_country, use_container_width=True)

    if 'TIPO' in df_filtrado.columns:
        tipo_counts = df_filtrado['TIPO'].value_counts().reset_index()
        tipo_counts.columns = ['TIPO', 'CANTIDAD']
        fig_tipo = px.bar(tipo_counts,
                          x='TIPO', y='CANTIDAD',
                          labels={'TIPO': 'Tipo de Vehículo', 'CANTIDAD': 'Cantidad'},
                          title='Distribución por Tipo de Vehículo')
        st.plotly_chart(fig_tipo, use_container_width=True)

    # PRONÓSTICO
    st.subheader("📈 Comparación de Pronósticos ARIMA, Prophet y ETS")
    ventas_ts = df_filtrado.groupby("MES_AÑO").size()
    ventas_ts.index = pd.to_datetime(ventas_ts.index)

    
    if len(ventas_ts) >= 18:
        train_data = ventas_ts[:-6]
        test_data = ventas_ts[-6:]

        forecast_arima = predecir_ventas(train_data, modelo='arima', pasos=6)
        forecast_prophet = predecir_ventas(train_data, modelo='prophet', pasos=6)
        forecast_ets = predecir_con_ets(train_data, pasos=6)

        forecast_arima = forecast_arima['ARIMA']
        forecast_prophet = forecast_prophet['Prophet']

        col3, col4 = st.columns([2, 1])

        with col3:
            fig_pred = Figure()
            # Serie de entrenamiento
            fig_pred.add_trace(Scatter(x=train_data.index, y=train_data.values, mode='lines', name='Entrenamiento'))

            # Serie real de test
            fig_pred.add_trace(Scatter(x=test_data.index, y=test_data.values, mode='lines', name='Real'))

            # Predicciones: asegúrate de que sean Series
            fig_pred.add_trace(Scatter(x=forecast_arima.index, y=forecast_arima.values, mode='lines', name='ARIMA'))
            fig_pred.add_trace(Scatter(x=forecast_prophet.index, y=forecast_prophet.values, mode='lines', name='Prophet'))
            fig_pred.add_trace(Scatter(x=forecast_ets.index, y=forecast_ets.values, mode='lines', name='ETS'))

            fig_pred.update_layout(title='Pronósticos de Ventas', xaxis_title='Fecha', yaxis_title='Ventas')

            st.plotly_chart(fig_pred, use_container_width=True)
        with col4:
            st.markdown("""
            ### 🧠 Métricas de Error:
            - **ARIMA**: MAE = {:.0f}, RMSE = {:.0f}  
            - **Prophet**: MAE = {:.0f}, RMSE = {:.0f}  
            - **ETS**: MAE = {:.0f}, RMSE = {:.0f}  
            """.format(
                mean_absolute_error(test_data, forecast_arima),
                np.sqrt(mean_squared_error(test_data, forecast_arima)),
                mean_absolute_error(test_data, forecast_prophet),
                np.sqrt(mean_squared_error(test_data, forecast_prophet)),
                mean_absolute_error(test_data, forecast_ets),
                np.sqrt(mean_squared_error(test_data, forecast_ets))
            ))
    else:
        st.warning("Datos insuficientes para entrenar modelos de pronóstico (requiere al menos 18 meses de datos).")

