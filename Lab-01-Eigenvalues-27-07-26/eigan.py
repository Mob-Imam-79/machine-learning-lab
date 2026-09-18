import math


def matrix_vector_multiply(A, x):
    n = len(A)
    result = [0.0] * n

    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result



def dot_product(a, b):
    s = 0.0
    for i in range(len(a)):
        s += a[i] * b[i]
    return s



def vector_norm(v):
    
    s = 0.0
    for x in v:
        s += x * x
    return math.sqrt(s)



def normalize(v):
    norm = vector_norm(v)
    if norm == 0:
        return v
    
    result = []

    for x in v:
        result.append(x / norm)

    return result



def copy_matrix(A):
    B = []
    for row in A:
        B.append(row[:])

    return B


def rayleigh_quotient(A, x):
    Ax = matrix_vector_multiply(A, x)
    numerator = dot_product(x, Ax)
    denominator = dot_product(x, x)

    if denominator == 0:
        return 0

    return numerator / denominator



def power_method(A, max_iterations=1000, tolerance=1e-8):

    n = len(A)
    x = [1.0] * n
    x = normalize(x)
    eigenvalue_old = 0.0

    for iteration in range(max_iterations):
        y = matrix_vector_multiply(A, x)
        x = normalize(y)
        eigenvalue = rayleigh_quotient(A, x)
        if abs(eigenvalue - eigenvalue_old) < tolerance:
            break
        eigenvalue_old = eigenvalue

    return eigenvalue, x


def outer_product(v1, v2):

    n = len(v1)
    result = []

    for i in range(n):
        row = []
        for j in range(n):
            row.append(v1[i] * v2[j])
        result.append(row)
    return result


def deflate_matrix(A, eigenvalue, eigenvector):
    n = len(A)
    new_matrix = copy_matrix(A)
    outer = outer_product(eigenvector, eigenvector)

    for i in range(n):
        for j in range(n):
            new_matrix[i][j] = new_matrix[i][j] - eigenvalue * outer[i][j]
    return new_matrix

def find_all_eigenpairs(A):

    n = len(A)
    matrix = copy_matrix(A)
    eigenvalues = []
    eigenvectors = []
    for i in range(n):
        eigenvalue, eigenvector = power_method(matrix)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        matrix = deflate_matrix(matrix, eigenvalue, eigenvector)

    return eigenvalues, eigenvectors

def sort_eigenpairs(eigenvalues, eigenvectors):
    n = len(eigenvalues)
    for i in range(n):

        max_index = i

        for j in range(i + 1, n):

            if eigenvalues[j] > eigenvalues[max_index]:
                max_index = j

        if max_index != i:

            temp = eigenvalues[i]
            eigenvalues[i] = eigenvalues[max_index]
            eigenvalues[max_index] = temp

            temp = eigenvectors[i]
            eigenvectors[i] = eigenvectors[max_index]
            eigenvectors[max_index] = temp

    return eigenvalues, eigenvectors


def get_principal_components(eigenvectors, k):

    principal_components = []

    for i in range(k):
        principal_components.append(eigenvectors[i])

    return principal_components



n = int(input("Enter order of matrix N: "))

A = []

print("Enter matrix elements row-wise:")

for i in range(n):
    row = list(map(float, input().split()))
    while len(row) != n:
        print("Please enter exactly", n, "values.")
        row = list(map(float, input().split()))
    A.append(row)

k = int(input("Enter value of K: "))

if k <= 0 or k > n:
    print("Invalid value of K!")
    print("K must be between 1 and", n)
    exit()

eigenvalues, eigenvectors = find_all_eigenpairs(A)

eigenvalues, eigenvectors = sort_eigenpairs(eigenvalues, eigenvectors)

principal_components = get_principal_components(eigenvectors, k)

print("\nEigenvalues")

for i in range(n):
    print("λ", i + 1, "=", round(eigenvalues[i], 6))

print("\nEigenvectors")

for i in range(n):
    print("\nEigenvector", i + 1)

    for value in eigenvectors[i]:
        print(round(value, 6))

print("\nTop", k, "Principal Components")

for i in range(k):

    print("\nPC", i + 1)

    for value in principal_components[i]:
        print(round(value, 6))


