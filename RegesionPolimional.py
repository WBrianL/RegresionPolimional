import numpy as np

# Datos de entrenamiento
x = np.array([
    [1, 41.9, 29.1],
    [1, 43.4, 29.3],
    [1, 43.9, 29.5],
    [1, 44.5, 29.7],
    [1, 47.3, 29.9],
    [1, 47.5, 30.3],
    [1, 47.9, 30.5],
    [1, 50.2, 30.7],
    [1, 52.2, 30.7],
    [1, 53.2, 30.9],
    [1, 56.7, 31.5],
    [1, 57.0, 31.7],
    [1, 63.5, 31.9],
    [1, 65.3, 32.0],
    [1, 71.1, 32.1],
    [1, 77.0, 32.5],
    [1, 77.8, 32.9]
])
y = np.array([
    251.3, 251.3, 248.3, 267.5, 273.0, 276.5, 270.3, 274.9,
    285.0, 290.0, 297.0, 302.5, 304.5, 309.3, 321.7, 330.7, 349.0
])

# Grado del polinomio
degree = 2

# Funcion para resolver un sistema de ecuaciones lineales utilizando eliminacion gaussiana
def solve_system(A, b):
    n = A.shape[0]
    for i in range(n):
        max_row = i
        for j in range(i + 1, n):
            if abs(A[j, i]) > abs(A[max_row, i]):
                max_row = j
        A[[i, max_row]] = A[[max_row, i]]
        b[[i, max_row]] = b[[max_row, i]]
        for j in range(i + 1, n):
            factor = A[j, i] / A[i, i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    coefficients = np.zeros(n)
    for i in range(n - 1, -1, -1):
        coefficients[i] = (b[i] - np.dot(A[i, i+1:], coefficients[i+1:])) / A[i, i]
    return coefficients

# Generar matriz X con potencias de x hasta el grado especificado
X = np.column_stack([x[:, 1]**i for i in range(degree + 1)])

# Resolver el sistema de ecuaciones para obtener los coeficientes del modelo
coefficients = solve_system(np.dot(X.T, X), np.dot(X.T, y))

# Calcular las predicciones del modelo para los datos de entrenamiento
predictions = np.dot(X, coefficients)

# Calcular el error cuadratico medio
mse = np.mean((predictions - y) ** 2)

# Imprimir los coeficientes del modelo
print("Coeficientes del modelo:", coefficients)

# Imprimir el error cuadratico medio
print("Error cuadratico medio:", mse)

# Variable para predecir un nuevo resultado
new_input = np.array([1, 60.0, 31.0])  # Cambia estos valores segun tu necesidad

# Predecir un nuevo resultado
new_prediction = np.dot(new_input, coefficients)
print("Prediccion para el nuevo dato:", new_prediction)
