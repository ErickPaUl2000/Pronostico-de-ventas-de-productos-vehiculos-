# utils_modelos.py

import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def predecir_ventas(serie, modelo='arima', pasos=6):
        """Genera predicciones usando ARIMA y Prophet"""
        resultados = {}

        # ARIMA
        if modelo == 'arima':
            modelo_arima = ARIMA(serie, order=(1, 1, 1))
            modelo_arima_fit = modelo_arima.fit()
            forecast_arima = modelo_arima_fit.forecast(steps=pasos)
            resultados['ARIMA'] = forecast_arima
        

        # Prophet
        elif modelo == 'prophet':
            df_prophet = serie.reset_index()
            df_prophet.columns = ['ds', 'y']
            modelo_prophet = Prophet()
            modelo_prophet.fit(df_prophet)
            future = modelo_prophet.make_future_dataframe(periods=pasos, freq='M')
            forecast_prophet = modelo_prophet.predict(future)
            resultados['Prophet'] = forecast_prophet.set_index('ds')['yhat'].iloc[-pasos:]
        
        else:
            raise ValueError("Modelo no reconocido. Usa 'arima' o 'prophet'.")

        return resultados

def predecir_con_ets(serie, pasos=6):
    """Predicción con ETS adaptativa según tamaño"""
    seasonal = 'add' if len(serie) >= 24 else None
    seasonal_periods = 12 if seasonal else None
    
    modelo_ets = ExponentialSmoothing(
        serie,
        trend='add',
        seasonal=seasonal,
        seasonal_periods=seasonal_periods
    )
    modelo_ets_fit = modelo_ets.fit()
    forecast_ets = modelo_ets_fit.forecast(pasos)
    return forecast_ets