import numpy as np

A = np.array([[3, -0.1, -0.2],
              [0.1, 7, -0.3],
              [0.3, -0.2, 10]])

b = np.array([7.85, -19.3, 71.4])


x = np.zeros(3)


tolerancia = 1e-6
max_iteraciones = 1000

def gauss_seidel(A, b, x, tolerancia, max_iteraciones):
    n = len(b)
    for iteracion in range(max_iteraciones):
        x_viejo = np.copy(x)

        for i in range(n):
            sigma = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

   
        error = np.linalg.norm(x - x_viejo, ord=np.inf)
        if error < tolerancia:
            break

    return x, iteracion + 1, error


solucion, iteraciones, error_final = gauss_seidel(A, b, x, tolerancia, max_iteraciones)


print(f"Solución: x1 = {solucion[0]:.6f}, x2 = {solucion[1]:.6f}, x3 = {solucion[2]:.6f}")
print(f"Número de iteraciones: {iteraciones}")
print(f"Error final: {error_final:.6e}")
