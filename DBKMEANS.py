import csv
import random
import time
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# Funzione per caricare i dati da un file CSV
def load_csv_data(file_path):
    data = []
    try:
        with open(file_path, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Salta la prima riga (intestazione)
            for row in reader:
                for value in row:
                    try:
                        # Prova a convertire ciascun valore in un intero
                        data.append(int(value))
                    except ValueError:
                        # Se non è un numero, lo ignora
                        continue
    except FileNotFoundError:
        print(f"Errore: il file {file_path} non è stato trovato.")
        return []
    
    return data

MIN_MERGE = 32

# Calcola il valore minimo del run per TimSort
def calc_min_run(n):
    """Calcola il valore minimo del run per TimSort.""" 
    r = 0
    while n >= MIN_MERGE:
        r |= n & 1
        n >>= 1
    return n + r

# Funzione di insertion sort
def insertion_sort(arr, left, right):
    """Ordina un array usando l'insertion sort.""" 
    for i in range(left + 1, right + 1):
        key_item = arr[i]
        j = i - 1
        while j >= left and arr[j] > key_item:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key_item

# Funzione di TimSort
def tim_sort(arr):
    """Ordina l'array usando l'algoritmo TimSort.""" 
    n = len(arr)
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
                merged_array = merge(arr[start:mid + 1], arr[mid + 1:end + 1])
                arr[start:start + len(merged_array)] = merged_array
        size *= 2

# Funzione per unire due array ordinati
def merge(left, right):
    """Unisce due array ordinati.""" 
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Funzione per applicare la strategia Alpha Stack
def apply_alpha_stack_strategy(alpha, stack):
    """Applica la strategia di alpha stacking.""" 
    if len(stack) < 2:
        return False

    Z = stack[-1]
    Y = stack[-2]

    # Verifica la condizione per l'alpha
    if len(Y) <= alpha * len(Z):
        # Esegui il merging
        stack.pop()  # Rimuovi Z
        stack.pop()  # Rimuovi Y

        merged_run = merge_runs(Y, Z)  # Funzione di merge
        stack.append(merged_run)  # Inserisci il risultato del merge
        return True

    return False

# Funzione di merge tra due run
def merge_runs(A, B):
    """Unisce due array ordinati"""
    out = []
    i = j = 0
    while i < len(A) and j < len(B):
        if A[i] <= B[j]:
            out.append(A[i])
            i += 1
        else:
            out.append(B[j])
            j += 1
    out.extend(A[i:])
    out.extend(B[j:])
    return out

# Funzione per eseguire l'Alpha Stack Sort
def alpha_stack_sort(arr, alpha=0.5):
    """Esegui l'alpha stack sort.""" 
    stack = [[x] for x in arr]  # Usa gli elementi di `arr` come singoli run iniziali
    while apply_alpha_stack_strategy(alpha, stack):
        pass  # Continua a eseguire finché la fusione è possibile

    print("Contenuto finale dello stack:", stack)
    return stack[-1]

# Funzione per eseguire il clustering con DBSCAN
def dbscan_clustering(data, eps=0.5, min_samples=5):
    """Esegui il clustering utilizzando DBSCAN.""" 
    # Normalizzare i dati prima del clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform([[x] for x in data])

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(data_scaled)
    
    # Crea i cluster separando i dati in base ai labels
    cluster_1 = [data[i] for i in range(len(data)) if labels[i] == 0]
    cluster_2 = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    return cluster_1, cluster_2, labels

# Funzione per eseguire il clustering con KMeans
def kmeans_clustering(data, n_clusters=2):
    """Esegui il clustering utilizzando KMeans.""" 
    # Normalizzare i dati prima del clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform([[x] for x in data])
    
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data_scaled)
    
    # Crea i cluster separando i dati in base ai labels
    cluster_1 = [data[i] for i in range(len(data)) if labels[i] == 0]
    cluster_2 = [data[i] for i in range(len(data)) if labels[i] == 1]
    
    return cluster_1, cluster_2, labels

# Funzione per misurare il tempo di esecuzione
def measure_execution_time(func, *args, **kwargs):
    """Misura il tempo di esecuzione di una funzione.""" 
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

# Funzione per visualizzare i cluster
def plot_clusters(cluster1, cluster2, title, cluster_time):
    """Crea un grafico dei cluster e stampa il tempo di clustering accanto al grafico.""" 
    plt.figure(figsize=(10, 5))

    # Disegna i due cluster
    plt.scatter(range(len(cluster1)), cluster1, color='r', label='Cluster 1', alpha=0.7)
    plt.scatter(range(len(cluster2)), cluster2, color='g', label='Cluster 2', alpha=0.7)

    # Titolo e etichette
    plt.title(title)
    plt.xlabel("Indice")
    plt.ylabel("Valore")
    plt.legend()
    plt.grid(True)

    # Aggiungi il tempo di esecuzione come testo nel grafico
    plt.text(0.05, 0.95, f"Tempo di clustering: {cluster_time:.4f} sec", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', horizontalalignment='left', color='black', weight='bold')
    plt.show()

# Esecuzione del programma principale
def main():
    # Parametri per DBSCAN e KMeans tramite input interattivi
    eps_value = float(input("Inserisci il valore di eps per DBSCAN (default 0.5): ") or 0.5)  # Impostato tramite input
    min_samples_value = int(input("Inserisci il valore di min_samples per DBSCAN (default 5): ") or 5)  # Impostato tramite input
    k_value = int(input("Inserisci il numero di cluster per KMeans (default 3): ") or 3)  # Impostato tramite input

    file_path = 'insurance.csv'  # Inserisci il percorso corretto del file CSV
    data = load_csv_data(file_path)
    
    if not data:
        print("Errore nel caricamento dei dati.")
        return

    USE_TIMSORT = True  # Imposta a True per usare TimSort, False per Alpha Stack Sort
    if USE_TIMSORT:
        print("Eseguendo TimSort...")
        sorted_data_tim = data.copy()  # Crea una copia dei dati originali
        _, tim_sort_time = measure_execution_time(tim_sort, sorted_data_tim)  # Misura il tempo di ordinamento in-place
        print("Dati ordinati con TimSort:", sorted_data_tim)

        # Misura il tempo di clustering con DBSCAN
        cluster_dbscan, dbscan_time = measure_execution_time(dbscan_clustering, sorted_data_tim, eps=eps_value, min_samples=min_samples_value)
        print("Cluster con DBSCAN:", cluster_dbscan)
        print(f"Tempo di esecuzione DBSCAN: {dbscan_time:.4f} secondi")
        
        # Visualizzazione dei cluster DBSCAN
        plot_clusters(cluster_dbscan[0], cluster_dbscan[1], "Cluster con DBSCAN", dbscan_time)

        # Misura il tempo di clustering con KMeans
        cluster_kmeans, kmeans_time = measure_execution_time(kmeans_clustering, sorted_data_tim, n_clusters=k_value)
        print("Cluster con KMeans:", cluster_kmeans)
        print(f"Tempo di esecuzione KMeans: {kmeans_time:.4f} secondi")
        
        # Visualizzazione dei cluster KMeans
        plot_clusters(cluster_kmeans[0], cluster_kmeans[1], "Cluster con KMeans", kmeans_time)

    else:
        print("Eseguendo Alpha Stack Sort...")
        sorted_data_alpha_stack, alpha_stack_time = measure_execution_time(alpha_stack_sort, data)
        print("Dati ordinati con Alpha Stack Sort:", sorted_data_alpha_stack)

        # Misura il tempo di clustering con DBSCAN
        cluster_dbscan, dbscan_time = measure_execution_time(dbscan_clustering, sorted_data_alpha_stack, eps=eps_value, min_samples=min_samples_value)
        print("Cluster con DBSCAN:", cluster_dbscan)
        print(f"Tempo di esecuzione DBSCAN: {dbscan_time:.4f} secondi")
        
        # Visualizzazione dei cluster DBSCAN
        plot_clusters(cluster_dbscan[0], cluster_dbscan[1], "Cluster con DBSCAN", dbscan_time)

        # Misura il tempo di clustering con KMeans
        cluster_kmeans, kmeans_time = measure_execution_time(kmeans_clustering, sorted_data_alpha_stack, n_clusters=k_value)
        print("Cluster con KMeans:", cluster_kmeans)
        print(f"Tempo di esecuzione KMeans: {kmeans_time:.4f} secondi")
        
        # Visualizzazione dei cluster KMeans
        plot_clusters(cluster_kmeans[0], cluster_kmeans[1], "Cluster con KMeans", kmeans_time)

# Esegui il programma principale
if __name__ == "__main__":
    main()
