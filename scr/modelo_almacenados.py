import joblib
import pandas as pd

# Cargar los modelos entrenados desde la ruta correcta
model_stock = joblib.load(r"C:\Users\franc\Desktop\Project_IA\ProyectoAzureIA\output\stock_model.pkl")
model_btc = joblib.load(r"C:\Users\franc\Desktop\Project_IA\ProyectoAzureIA\output\bitcoin_model.pkl")

# Predicciones para SPY (Bolsa de valores)
new_data_spy = pd.DataFrame({
    'Open': [350.50],
    'High': [355.00],
    'Low': [348.50],
    'Volume': [1500000],
    'Moving_Avg': [352.00]  # Debes asegurarte de calcular el promedio móvil en los datos nuevos
})

spy_pred = model_stock.predict(new_data_spy)
print("Predicción para SPY (sube o baja):", "Sube" if spy_pred[0] else "Baja")

# Predicciones para Bitcoin (Valor futuro)
new_data_btc = pd.DataFrame({
    'Open': [33500],
    'High': [34000],
    'Low': [33000],
    'Volume': [1000000]
})

# Renombrar las columnas para que coincidan con las del modelo entrenado
new_data_btc.rename(
    columns={
        'Open': 'Open BTC-USD',
        'High': 'High BTC-USD',
        'Low': 'Low BTC-USD',
        'Volume': 'Volume BTC-USD'
    },
    inplace=True
)

btc_pred = model_btc.predict(new_data_btc)
print("Predicción para el valor de Bitcoin:", btc_pred[0])
