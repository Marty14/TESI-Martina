import csv
import random
import time

# Funzione per caricare i dati da un file CSV
def load_csv_data(file_path):
    data = []
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

# Funzione per calcolare la dimensione del run
def get_run_size(run):
    return len(run)

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

# Funzione che applica la strategia di alpha stacking
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

# Funzione per eseguire l'alpha stack sort
def alpha_stack_sort(arr, alpha=0.5):
    """Esegui l'alpha stack sort."""
    stack = [[1, 3, 5], [2, 4, 6]]  # Esempio di stack iniziale
    while apply_alpha_stack_strategy(alpha, stack):
        pass  # Continua a eseguire finché la fusione è possibile

    # Stampa il contenuto finale dello stack
    print("Contenuto finale dello stack:", stack)
    return stack[-1]

# Funzione per eseguire l'advanced stack sort (non utilizzato in questo esempio)
def advanced_stack_sort(arr, use_timsort=False):
    """Esegui l'advanced stack sort."""
    comparison_count = 0
    EXTRA_RAND_NUM_COUNT = int(1e4)

    def print_runs(runs):
        for run in runs:
            print("[", " ".join(map(str, run)), "]", end=" ")
        print()

    # Aggiungi un po' di numeri casuali per testare
    sequence = [2, 3, 5, 4, 7, 4, 1, 11, 11, 10, 9, 8, 7, 9, 10, 2]
    sequence += [random.randint(1, 100) for _ in range(EXTRA_RAND_NUM_COUNT)]

    run_decomposition = create_run_decomposition(sequence)
    stack = []

    while run_decomposition:
        run = run_decomposition.pop()
        stack.append(run)

        while apply_alpha_stack_strategy(2, stack):
            pass

    while len(stack) > 1:
        Z = stack.pop()
        Y = stack.pop()
        merged_run = merge_runs(Z, Y)
        stack.append(merged_run)

    print("Dati ordinati con Advanced Stack Sort:", stack)

# Funzione per creare la decomposizione in run
def create_run_decomposition(sequence):
    run_decomposition = []
    if not sequence:
        return run_decomposition

    UNKNOWN = 0
    ASCENDING = 1
    DESCENDING = 2
    direction = UNKNOWN

    current_run = [sequence[0]]
    
    for i in range(1, len(sequence)):
        current_elem = sequence[i]
        prev_elem = sequence[i - 1]

        if current_elem != prev_elem:
            if direction == UNKNOWN:
                direction = ASCENDING if current_elem > prev_elem else DESCENDING
            elif (direction == ASCENDING and current_elem < prev_elem) or (direction == DESCENDING and current_elem > prev_elem):
                if direction == DESCENDING:
                    current_run.reverse()
                run_decomposition.append(current_run)
                current_run = []
                direction = UNKNOWN

        current_run.append(current_elem)

        if i == len(sequence) - 1:
            if direction == DESCENDING:
                current_run.reverse()
            run_decomposition.append(current_run)

    return run_decomposition

# Funzione per eseguire il clustering dei dati ordinati
def cluster_data(sorted_data):
    """Esegui il clustering dei dati ordinati."""
    median = sorted_data[len(sorted_data) // 2]
    cluster_below_median = [x for x in sorted_data if x <= median]
    cluster_above_median = [x for x in sorted_data if x > median]
    
    return cluster_below_median, cluster_above_median

#Funzione per misurare il tempo di esecuzione
def measure_execution_time(func, *args, **kwargs):
    """Misura il tempo di esecuzione di una funzione."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    return result, execution_time

# Funzione main per testare tutto il codice
def main():
    file_path = r"C:\Users\bmart\OneDrive\Desktop\UNI\TESI\insurance.csv"
    data = load_csv_data(file_path)
    
    print("Dati originali:", data)

    # Imposta quale algoritmo usare: True per TimSort, False per Alpha Stack Sort
    USE_TIMSORT = True  # Cambia a False per Alpha Stack Sort

    if USE_TIMSORT:
        print("Eseguendo TimSort...")
    sorted_data_tim = data.copy()  # Crea una copia dei dati originali
    _, tim_sort_time = measure_execution_time(tim_sort, sorted_data_tim)  # Misura il tempo di ordinamento in-place
    print("Dati ordinati con TimSort:", sorted_data_tim)

    # Misura il tempo di clustering
    cluster_tim, cluster_time = measure_execution_time(cluster_data, sorted_data_tim)
    print("Cluster con TimSort:", cluster_tim)
    print(f"Tempo di esecuzione TimSort: {tim_sort_time:.4f} secondi")
    print(f"Tempo di esecuzione clustering: {cluster_time:.4f} secondi")


    if not USE_TIMSORT:
        print("Eseguendo Alpha Stack Sort...")
    sorted_data_alpha, alpha_sort_time = measure_execution_time(alpha_stack_sort, data.copy())
    print("Dati ordinati con Alpha Stack Sort:", sorted_data_alpha)

    cluster_alpha, cluster_time = measure_execution_time(cluster_data, sorted_data_alpha)
    print("Cluster con Alpha Stack Sort:", cluster_alpha)
    print(f"Tempo di esecuzione Alpha Stack Sort: {alpha_sort_time:.4f} secondi")
    print(f"Tempo di esecuzione clustering: {cluster_time:.4f} secondi")


    # Se vuoi aggiungere un'opzione per Advanced Stack Sort, puoi aggiungerla qui
    USE_ADVANCED_STACK_SORT = False  # Cambia a True per usare Advanced Stack Sort
    if USE_ADVANCED_STACK_SORT:
        print("Eseguendo Advanced Stack Sort...")
        sorted_data_advanced = data.copy()
        advanced_stack_sort(sorted_data_advanced)
        print("Dati ordinati con Advanced Stack Sort:", sorted_data_advanced)
        cluster_advanced = cluster_data(sorted_data_advanced)
        print("Cluster con Advanced Stack Sort:", cluster_advanced)

if __name__ == "__main__":
    main()
