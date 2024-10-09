import numpy as np
import statsmodels.api as sm
import time
import matplotlib.pyplot as plt  # Diesen Import hinzufügen

# Lösen eines LGS mit Dreiecksmatrix durch rückwärts einsetzen
def backward_substitution(A, y):
    n = A.shape[0]  # Dimension der Dreiecksmatrix bestimmen
    b = np.zeros(n)  # Lösung für den Vektor b

    print(n)

    for i in range(n - 1, -1, -1):
        if A[i, i] == 0:
            raise ValueError("Die Matrix A enthält eine Null auf der Diagonale, keine eindeutige Lösung möglich.")
        
        b[i] = (y[i] - np.dot(A[i, i+1:], b[i+1:])) / A[i, i]

    return b

# Manuelle QR-Zerlegung (Gram-Schmidt-Verfahren)
def gram_schmidt(X):
    n, m = X.shape
    Q = np.zeros((n, m))
    R = np.zeros((m, m))

    for j in range(m):
        v = X[:, j]
        
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], X[:, j])
            v = v - R[i, j] * Q[:, i]

        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]

    return Q, R

# Ausgedachter Datensatz
def create_data(n, p, beta_true):
    np.random.seed(42)
    X = np.random.rand(n, p)
    X = np.column_stack([np.ones(X.shape[0]), X])
    y = X @ beta_true + np.random.randn(n) * 0.1
    return X, y

# Funktion für lineare Regression mittels QR Zerlegung mit Zeitmessung
def linear_regression_manual_qr(X, y):
    start_time = time.time()
    Q, R = gram_schmidt(X)
    beta = backward_substitution(R, Q.T @ y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return beta, elapsed_time

# Funktion für die Durchführung der Regression für unterschiedliche n-Werte mit mehrfachen Wiederholungen
def run_benchmark(n_list, repetitions=5):
    results = []
    beta_true = [-8, -1.6, 4.1, -10, -9.2, 1.3, 1.6, 2.3]
    p = 7

    for n in n_list:
        times = []
        for _ in range(repetitions):
            X, y = create_data(n, p, beta_true)
            beta, elapsed_time = linear_regression_manual_qr(X, y)
            times.append(elapsed_time)

        avg_time = np.mean(times)
        std_time = np.std(times)
        results.append([n, avg_time, std_time, beta])

    return results

# Durchführung des Benchmarks
n_values = [100, 500, 1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
benchmark_results = run_benchmark(n_values)

# Ausgeben der Ergebnisse
for i in benchmark_results:
    print("\n Datenzeilen: ", i[0])
    print("Laufzeit: ", i[1])
    #print("Berchnete Parameter: ", i[3])

# Grafische Darstellung der Ergebnisse
avg_times = [result[1] for result in benchmark_results]
std_times = [result[2] for result in benchmark_results]

plt.figure(figsize=(10, 6))
plt.errorbar(n_values, avg_times, yerr=std_times, fmt='-o', ecolor='r', capsize=5, label='Durchschnittliche Rechenzeit mit Standardabweichung')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Anzahl der Beobachtungen (n)')
plt.ylabel('Rechenzeit (s)')
plt.title('Rechenzeit der QR-Dekompositions-basierten Regression für wachsendes n')
plt.legend()
plt.grid(True)
plt.show()

'''
# Vergleich mit statsmodels
X, y = create_data(1000, 7, [-8, -1.6, 4.1, -10, -9.2, 1.3, 1.6, 2.3])
model = sm.OLS(y, X)
results = model.fit()

print("\nStatsmodels OLS Ergebnisse:\n", results.params)
beta_manual, _ = linear_regression_manual_qr(X, y)
print("\nManuell berechnete Koeffizienten:\n", beta_manual)
'''