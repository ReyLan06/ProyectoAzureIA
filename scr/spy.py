from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import joblib

# Cargar el modelo entrenado
model_stock = joblib.load(r"C:\Users\franc\Desktop\Project_IA\ProyectoAzureIA\output\stock_model.pkl")

# Crear datos de prueba simulados (reemplaza esto con un archivo CSV si lo tienes)
test_data = pd.DataFrame({
    'Open': [350.5, 340.0, 360.0],
    'High': [355.0, 345.0, 365.0],
    'Low': [348.5, 338.0, 358.0],
    'Volume': [1500000, 1600000, 1700000],
    'Moving_Avg': [352.0, 342.0, 362.0],  # Característica calculada
    'Target': [1, 0, 1]  # Etiquetas reales
})


X_test = test_data[['Open', 'High', 'Low', 'Volume', 'Moving_Avg']]
y_test = test_data['Target']

# Hacer predicciones con el modelo cargado
predictions = model_stock.predict(X_test)

# Evaluar precisión
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión del modelo: {accuracy}")

plt.plot(y_test.values, label="Real")
plt.plot(predictions, label="Predicho")
plt.legend()
plt.show()