import matplotlib.pyplot as plt

plt.plot(y_test.values, label="Real")
plt.plot(predictions, label="Predicho")
plt.legend()
plt.show()