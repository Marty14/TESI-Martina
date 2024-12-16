import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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

# Funzione di ordinamento TimSort
def tim_sort(arr):
    MIN_MERGE = 32
    comparisons = 0

    def calc_min_run(n):
        r = 0
        while n >= MIN_MERGE:
            r |= n & 1
            n >>= 1
        return n + r

    def insertion_sort(arr, left, right):
        nonlocal comparisons
        for i in range(left + 1, right + 1):
            key_item = arr[i]
            j = i - 1
            while j >= left and tuple(arr[j]) > tuple(key_item):
                comparisons += 1
                arr[j + 1] = arr[j]
                j -= 1
                comparisons += 1
            arr[j + 1] = key_item

    def merge(left, right):
        nonlocal comparisons
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            comparisons += 1
            if tuple(left[i]) <= tuple(right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result

    n = len(arr)
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
                left = arr[start:mid + 1]
                right = arr[mid + 1:end + 1]
                arr[start:start + len(left) + len(right)] = merge(left, right)
        size *= 2

    return arr, comparisons

# Funzione di ordinamento Alpha Stack Sort con conteggio confronti
def alpha_stack_sort(data):
    comparisons = 0
    data_as_tuples = [tuple(row) for row in data]
    stack = []
    for value in data_as_tuples:
        while stack and stack[-1] > value:
            comparisons += 1
            stack.pop()
            comparisons += 1  # Confronto fallito o valore inserito
        stack.append(value)
    return np.array(sorted(stack)), comparisons

# Clustering con DBSCAN
def dbscan_clustering(data, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_scaled)
    return labels

# Clustering con KMeans
def kmeans_clustering(data, n_clusters=2):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_scaled)
    return labels

# Ricerca dei migliori parametri per KMeans (K)
def find_best_kmeans_k(data, min_k=2, max_k=10):
    best_k = min_k
    best_silhouette = -1
    best_range = (min_k, max_k)

    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        labels = kmeans.fit_predict(data)
        if len(set(labels)) > 1:  # Assicurarsi che ci siano almeno 2 cluster
            silhouette_avg = silhouette_score(data, labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_k = k
                best_range = (min_k, max_k)

    print(f"\nMiglior valore di K per K-Means: {best_k}, con Silhouette Score: {best_silhouette:.4f}")
    print(f"Range testato: {best_range}")
    return best_k, best_silhouette

# Ricerca dei migliori parametri per DBSCAN (eps)
def find_best_dbscan_eps(data, min_eps=0.1, max_eps=2.0, step=0.1):
    best_eps = min_eps
    best_silhouette = -1
    best_range = (min_eps, max_eps)

    for eps in np.arange(min_eps, max_eps, step):
        db = DBSCAN(eps=eps, min_samples=5)
        labels = db.fit_predict(data)
        if len(set(labels)) > 1:  # Assicurarsi che ci siano almeno 2 cluster
            silhouette_avg = silhouette_score(data, labels)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_eps = eps
                best_range = (min_eps, max_eps)

    print(f"\nMiglior valore di eps per DBSCAN: {best_eps}, con Silhouette Score: {best_silhouette:.4f}")
    print(f"Range testato: {best_range}")
    return best_eps, best_silhouette

# Clustering con CLASSIX
class CLASSIX:
    def __init__(self, sorting_func):
        self.sorting_func = sorting_func

    def fit_predict(self, data):
        data_sorted, _ = self.sorting_func(data)
        clusters = self._cluster_data(data_sorted)
        return clusters

    def _cluster_data(self, data):
        return np.random.randint(0, 3, size=len(data))

# Misura del tempo di esecuzione
def measure_execution_time(func, *args, **kwargs):
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time

# Visualizzazione dei cluster
def plot_clusters(data, labels, title, clustering_time):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.title(f"{title} (Tempo: {clustering_time:.4f} sec)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Cluster')
    plt.show()

# Funzione principale
def main():
    file_path = "insurance.csv"  
    # Pre-elabora i dati
    data = preprocess_data(file_path)

    # Trova il miglior K per KMeans
    best_k, best_k_score = find_best_kmeans_k(data)
    
    # Trova il miglior eps per DBSCAN
    best_eps, best_eps_score = find_best_dbscan_eps(data)


    # Liste per memorizzare i risultati dei tempi di esecuzione e dei silhouette score
    sorting_algorithms = [
        ("TimSort", tim_sort),
        ("Alpha Stack Sort", alpha_stack_sort)
    ]

    clustering_methods = [
        ("DBSCAN", lambda data: dbscan_clustering(data, eps=best_eps)),
        ("KMeans", lambda data: kmeans_clustering(data, n_clusters=best_k)),
        ("CLASSIX", lambda data: CLASSIX(sorting_func=tim_sort).fit_predict(data))  # Usa TimSort per CLASSIX
    ]
    time_results = []
    silhouette_results = []
    comparison_results = []

    for algo_name, sorting_func in sorting_algorithms:
        print(f"\nOrdinamento: {algo_name}")
        sorted_data, sort_time, comparisons = measure_execution_time_with_comparisons(sorting_func, data)
        time_results.append((algo_name, sort_time))  # Memorizza i tempi di ordinamento
        comparison_results.append((algo_name, comparisons))

        for cluster_name, cluster_func in clustering_methods:
            print(f"Clustering: {cluster_name}")
            labels, cluster_time = measure_execution_time(cluster_func, sorted_data)
            total_time = sort_time + cluster_time  # Tempo totale = ordinamento + clustering
            time_results.append((f"{algo_name} + {cluster_name}", total_time))  # Memorizza i tempi di esecuzione complessivi

            # Controllo consistenza dimensioni
            if len(labels) != len(sorted_data):
                print(f"Errore: il numero di etichette ({len(labels)}) non corrisponde al numero di campioni ({len(sorted_data)}).")
                continue

            print(f"Clustering: {cluster_name} | Tempo di clustering: {cluster_time:.4f} sec")

            # Filtra i dati per calcolare il silhouette score solo sui campioni validi
            valid_indices = labels != -1  # Considera i punti con etichette valide
            filtered_sorted_data = sorted_data[valid_indices]
            filtered_labels = labels[valid_indices]
            
            if len(set(filtered_labels)) > 1:  # Almeno 2 cluster
                silhouette_avg = silhouette_score(filtered_sorted_data, filtered_labels)
                silhouette_results.append((f"{algo_name} + {cluster_name}", silhouette_avg))  # Memorizza il silhouette score
                print(f"Silhouette Score: {silhouette_avg:.4f}")
            else:
                silhouette_results.append((f"{algo_name} + {cluster_name}", None))  # Nessun Silhouette Score
                print("Silhouette Score non calcolabile (numero di cluster < 2).")

            plot_clusters(sorted_data, labels, f"{algo_name} + {cluster_name}", cluster_time)

# Plot dei Tempi di Esecuzione
    labels, times = zip(*time_results)
    plt.figure(figsize=(12, 6))
    plt.barh(labels, times, color='skyblue')
    plt.xlabel('Tempo di Esecuzione (secondi)')
    plt.title('Confronto Tempi di Esecuzione per Ordinamento e Clustering')
    plt.show()

# Plot dei Silhouette Score
    valid_scores = [(label, score) for label, score in silhouette_results if score is not None]
    if valid_scores:
        labels, scores = zip(*valid_scores)
        plt.figure(figsize=(12, 6))
        plt.barh(labels, scores, color='lightgreen')
        plt.xlabel('Silhouette Score')
        plt.title('Confronto Silhouette Score per Ordinamento e Clustering')
        plt.show()

# Plot del Numero di Confronti
    labels, comparisons = zip(*comparison_results)
    plt.figure(figsize=(12, 6))
    plt.barh(labels, comparisons, color='lightcoral')
    plt.xlabel('Numero di Confronti')
    plt.title('Confronto Numero di Confronti per Algoritmo di Ordinamento')
    plt.show()

# Misura del tempo e dei confronti per l'ordinamento
def measure_execution_time_with_comparisons(sorting_func, *args, **kwargs):
    start_time = time.perf_counter()
    result, comparisons = sorting_func(*args, **kwargs)
    end_time = time.perf_counter()
    return result, end_time - start_time, comparisons

if __name__ == "__main__":
    main()