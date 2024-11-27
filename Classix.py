import numpy as np
import pandas as pd
import time

# Implementazione di SimpleStandardScaler
class SimpleStandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        """
        Calcola la media e la deviazione standard per ciascuna colonna.
        """
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        """
        Applica la standardizzazione ai dati.
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Lo scaler non è stato ancora adattato ai dati. Chiama il metodo fit().")
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        """
        Combina i passaggi di fit e transform.
        """
        return self.fit(X).transform(X)

# Funzione per preprocessare i dati
def preprocess_data(file_path):
    """
    Pre-elabora i dati, standardizzando quelli numerici e codificando quelli categoriali.
    """
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
        """
        Esegue il clustering sui dati.
        """
        data_sorted = self.sorting_func(data)
        if data_sorted is None:
            raise ValueError("L'array ordinato non è valido.")
        clusters = self._cluster_data(data_sorted)
        return clusters

    def _cluster_data(self, data):
        """
        Implementazione base del clustering (mock).
        """
        return np.random.randint(0, 3, size=len(data))  # Cluster casuali (esempio)

# Funzioni di ordinamento e conteggio dei confronti
comparison_count = 0  # Contatore globale

def insertion_sort(arr, left, right):
    """Ordina un array usando l'insertion sort e conta i confronti."""
    global comparison_count
    for i in range(left + 1, right + 1):
        key_item = arr[i]
        j = i - 1
        # Confronta le righe come tuple (o una colonna specifica, es. arr[j][0] se vuoi solo una colonna)
        while j >= left and tuple(arr[j]) > tuple(key_item):  # Confronta le righe come tuple
            comparison_count += 1  # Incrementa il contatore
            arr[j + 1] = arr[j]
            j -= 1
        if j >= left:
            comparison_count += 1  # Confronto finale falso
        arr[j + 1] = key_item

def merge(left, right):
    """Unisce due array ordinati, considerando le righe come tuple."""
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        # Confronta le righe come tuple per evitare problemi di confronto
        if tuple(left[i]) <= tuple(right[j]):  # Confronta le righe come tuple
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    # Aggiungi gli elementi rimanenti
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def tim_sort(arr):
    """Ordina l'array usando l'algoritmo TimSort."""
    global comparison_count
    n = len(arr)
    MIN_MERGE = 32

    def calc_min_run(n):
        """Calcola il valore minimo del run per TimSort."""
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def merge_runs(start, mid, end):
        """Unisce due run."""
        left = arr[start:mid + 1]
        right = arr[mid + 1:end + 1]
        merged = merge(left, right)
        arr[start:start + len(merged)] = merged

    # Debug: Verifica se l'array è valido prima di ordinare
    print(f"Array iniziale (prima di TimSort): {arr[:10]}...")  # Solo i primi 10 elementi per visualizzare

    min_run = calc_min_run(n)

    # Esegui insertion sort su segmenti di lunghezza min_run
    for start in range(0, n, min_run):
        end = min(start + min_run - 1, n - 1)
        insertion_sort(arr, start, end)

    # Unisci i segmenti
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            mid = start + size - 1
            end = min((start + 2 * size - 1), (n - 1))
            if mid < end:  # Verifica che ci siano due sezioni da unire
                merge_runs(start, mid, end)
        size *= 2

    # Debug: Verifica se l'array ordinato è valido
    print(f"Array ordinato (dopo TimSort): {arr[:10]}...")  # Solo i primi 10 elementi per visualizzare
    return arr

def fit_predict(self, data):
    """
    Esegue il clustering sui dati.
    """
    # Debug: Verifica se i dati sono validi prima dell'ordinamento
    if data is None or len(data) == 0:
        raise ValueError("I dati di input non sono validi.")
    
    data_sorted = self.sorting_func(data)
    
    # Debug: Verifica che i dati ordinati siano validi
    if data_sorted is None or len(data_sorted) == 0:
        raise ValueError("L'array ordinato non è valido.")
    
    clusters = self._cluster_data(data_sorted)
    return clusters

def alpha_stack_sort(data):
    """
    Alpha Stack Sort per array numpy, con conteggio dei confronti.
    """
    global comparison_count
    data_as_tuples = [tuple(row) for row in data]  # Converti le righe in tuple
    stack = []

    for value in data_as_tuples:
        while stack and stack[-1] > value:
            comparison_count += 1  # Incrementa il contatore
            stack.pop()
        if stack and stack[-1] <= value:
            comparison_count += 1  # Confronto finale falso
        stack.append(value)

    return np.array(sorted(stack))  # Torna all'array numpy ordinato

# Funzione per misurare il tempo di esecuzione
def measure_execution_time(func, *args, **kwargs):
    """Misura il tempo di esecuzione di una funzione."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

## Funzione principale
def main():
    file_path = "insurance.csv"  # Cambia con il tuo file

    # Pre-elabora i dati
    data = preprocess_data(file_path)

    # CLASSIX senza ordinamento
    global comparison_count
    comparison_count = 0
    print("\n--- CLASSIX senza ordinamento ---")
    classix_no_sort = CLASSIX(sorting_func=lambda x: x)  # Passa una funzione che restituisce i dati senza modificarli
    _, classix_no_sort_time = measure_execution_time(classix_no_sort.fit_predict, data)
    print(f"Tempo di esecuzione CLASSIX senza ordinamento: {classix_no_sort_time:.4f} secondi")
    print(f"Numero di confronti in CLASSIX senza ordinamento: {comparison_count}")

    # CLASSIX con TimSort
    comparison_count = 0
    print("\n--- CLASSIX con TimSort ---")
    classix_tim = CLASSIX(sorting_func=tim_sort)
    _, tim_sort_time = measure_execution_time(classix_tim.fit_predict, data)
    print(f"Tempo di esecuzione CLASSIX con TimSort: {tim_sort_time:.4f} secondi")
    print(f"Numero di confronti in TimSort: {comparison_count}")

    # CLASSIX con Alpha Stack Sort
    comparison_count = 0
    print("\n--- CLASSIX con Alpha Stack Sort ---")
    classix_alpha = CLASSIX(sorting_func=alpha_stack_sort)
    _, alpha_stack_time = measure_execution_time(classix_alpha.fit_predict, data)
    print(f"Tempo di esecuzione CLASSIX con Alpha Stack Sort: {alpha_stack_time:.4f} secondi")
    print(f"Numero di confronti in Alpha Stack Sort: {comparison_count}")

if __name__ == "__main__":
    main()


# 1. Differenza nei risultati tra TimSort e Alpha Stack Sort:
#- TimSort:
#    - Restituisce un array ordinato, ma con tutti gli elementi identici
#- Alpha Stack Sort:
 #   - Restituisce un array ordinato, ma solo con due risultati distinti. Questo probabilmente succede perché l'algoritmo sta facendo un ordinamento che considera solo alcune colonne (o una parte dell'array), o c'è qualche errore nel modo in cui gestisce il confronto tra gli elementi.
