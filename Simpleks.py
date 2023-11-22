import numpy as np

def simplex_method(c, A, b):
    # Inisialisasi variabel
    #m mewakili banyak baris dari A
    #n mewakili banyak kolom dari A
    m, n = A.shape
    
    #membuat matriks kosong
    #m=2,n=2
    #baris = 3 dan kolom = 5
    # [[0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]
    # [0. 0. 0. 0. 0.]]
    matrikz = np.zeros((m + 1, m + n + 1))
    
    # Inisialisasi variabel basis
    # array = np.arange(2, 2+2)    
    # Output: [2 3]
    basis = np.arange(n, n + m)
    
    # Mengisi matriks
    matrikz[0, :n] = -c #mengisi baris pertama(Z)
    matrikz[1:, :n] = A #mengisi baris kedua(S1,S2,...Sn)
    matrikz[1:, -1] = b #mengisi kolom terakhir(RK)

    iteration = 0
    while True:
        print(f"### Iterasi {iteration} ###")
        # Menampilkan tabel Simpleks pada setiap iterasi
        print(matrikz)

        # mencari nilai kolom dengan koefisien negatif terbesar dibaris pertama
        pivot_col = np.argmax(matrikz[0, :-1])

        # Jika koefisien dalam baris pertama adalah non-negatif, 
        # itu berarti tidak ada kemungkinan peningkatan lebih lanjut 
        # dalam nilai fungsi tujuan. Dengan kata lain, sudah 
        # mencapai solusi yang optimal  
        if matrikz[0, pivot_col] <= 0:
            # Solusi optimal ditemukan
            break

        # Cari indeks tiap baris 
        indeks = matrikz[1:, -1] / matrikz[1:, pivot_col]
        pivot_row = np.argmin(indeks) + 1 

        # Lakukan operasi baris untuk membuat elemen pivot menjadi 1
        pivot_element = matrikz[pivot_row, pivot_col]
        matrikz[pivot_row, :] /= pivot_element

        # Lakukan operasi baris lainnya untuk membuat elemen lain di kolom pivot menjadi 0
        for i in range(m + 1):
            if i != pivot_row:
                ratio = matrikz[i, pivot_col]
                matrikz[i, :] -= ratio * matrikz[pivot_row, :]

        # Perbarui basis
        basis[pivot_row - 1] = pivot_col

        iteration += 1

    # Ekstrak solusi dari tabel
    solution = np.zeros(n)
    for i in range(m):
        if basis[i] < n:
            solution[basis[i]] = matrikz[i + 1, -1]

    optimal_value = -matrikz[0, -1]

    return solution, optimal_value

# Contoh masalah pemrograman linear
c = np.array([-8, -6])
A = np.array([[4, 2],
              [2, 4]])
b = np.array([60, 48])

solution, optimal_value = simplex_method(c, A, b)
print("Solusi optimal:")
print("X1 =", solution[0])
print("X2 =", solution[1])
print("Nilai maksimum Z =", optimal_value)
