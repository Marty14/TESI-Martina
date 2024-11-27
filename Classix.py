import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt  # Importa matplotlib per il grafico

# Implementazione di SimpleStandardScaler
class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Lo scaler non è stato ancora adattato ai dati. Chiama il metodo fit().")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# Funzione per preprocessare i dati
def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    non_numeric_columns = data.select_dtypes(include=['object']).columns
    for col in non_numeric_columns:
        data[col] = pd.factorize(data[col])[0]

    scaler = SimpleStandardScaler()
    scaled_data = scaler.fit_transform(data.values)
    print("Dati preprocessati con successo!")
    return scaled_data

# Funzione CLASSIX
class CLASSIX:
    def __init__(self, sorting_func):
        self.sorting_func = sorting_func

    def fit_predict(self, data):
        data_sorted = self.sorting_func(data)
        if data_sorted is None:
            raise ValueError("L'array ordinato non è valido.")
        clusters = self._cluster_data(data_sorted)
        return clusters

    def _cluster_data(self, data):
        return np.random.randint(0, 3, size=len(data))

# Funzioni di ordinamento
comparison_count = 0  # Contatore globale

def insertion_sort(arr, left, right):
    global comparison_count
    for i in range(left + 1, right + 1):
        key_item = arr[i]
        j = i - 1
        while j >= left and tuple(arr[j]) > tuple(key_item):
            comparison_count += 1
            arr[j + 1] = arr[j]
            j -= 1
        if j >= left:
            comparison_count += 1
        arr[j + 1] = key_item

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if tuple(left[i]) <= tuple(right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def tim_sort(arr):
    global comparison_count
    n = len(arr)
    MIN_MERGE = 32

    def calc_min_run(n):
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def merge_runs(start, mid, end):
        left = arr[start:mid + 1]
        right = arr[mid + 1:end + 1]
        merged = merge(left, right)
        arr[start:start + len(merged)] = merged

    min_run = calc_min_run(n)

    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        insertion_sort(arr, start, end)

    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min((start + 2 * size - 1), (n - 1))
            if mid < end:
                merge_runs(start, mid, end)
        size *= 2

    return arr

def alpha_stack_sort(data):
    global comparison_count
    data_as_tuples = [tuple(row) for row in data]
    stack = []
    for value in data_as_tuples:
        while stack and stack[-1] > value:
            comparison_count += 1
            stack.pop()
        if stack and stack[-1] <= value:
            comparison_count += 1
        stack.append(value)

    return np.array(sorted(stack))

def measure_execution_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

# Funzione principale
def main():
    file_path = "insurance.csv"  # Cambia con il tuo file

    # Pre-elabora i dati
    data = preprocess_data(file_path)

    # Classificazioni e misurazioni del tempo
    results = []

    # CLASSIX senza ordinamento
    global comparison_count
    comparison_count = 0
    classix_no_sort = CLASSIX(sorting_func=lambda x: x)
    _, classix_no_sort_time = measure_execution_time(classix_no_sort.fit_predict, data)
    results.append(("Senza ordinamento", classix_no_sort_time))

    # CLASSIX con TimSort
    comparison_count = 0
    classix_tim = CLASSIX(sorting_func=tim_sort)
    _, tim_sort_time = measure_execution_time(classix_tim.fit_predict, data)
    results.append(("TimSort", tim_sort_time))

    # CLASSIX con Alpha Stack Sort
    comparison_count = 0
    classix_alpha = CLASSIX(sorting_func=alpha_stack_sort)
    _, alpha_stack_time = measure_execution_time(classix_alpha.fit_predict, data)
    results.append(("Alpha Stack Sort", alpha_stack_time))

    # Stampa dei risultati
    for name, exec_time in results:
        print(f"Tempo di esecuzione {name}: {exec_time:.4f} secondi")

    # Visualizzazione dei tempi di esecuzione
    labels = [name for name, _ in results]
    times = [exec_time for _, exec_time in results]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, times, color=['skyblue', 'lightgreen', 'salmon'])
    plt.xlabel("Algoritmo")
    plt.ylabel("Tempo di esecuzione (secondi)")
    plt.title("Confronto dei tempi di esecuzione dei vari algoritmi di ordinamento")
    plt.show()

if __name__ == "__main__":
    main()
