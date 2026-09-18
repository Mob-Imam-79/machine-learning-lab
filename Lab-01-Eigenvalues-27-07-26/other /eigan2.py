import math


def identity_matrix(n):
    I = []
    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                row.append(1.0)
            else:
                row.append(0.0)
        I.append(row)
    return I

def max_offdiag(A):
    n = len(A)
    max_val = 0.0
    p = 0
    q = 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(A[i][j]) > max_val:
                max_val = abs(A[i][j])
                p = i
                q = j

    return max_val, p, q


def jacobi(A):
    n = len(A)
    V = identity_matrix(n)

    max_iterations = 100

    for _ in range(max_iterations):

        max_val, p, q = max_offdiag(A)

        if max_val < 1e-10:
            break

        if A[p][p] == A[q][q]:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan((2 * A[p][q]) / (A[q][q] - A[p][p]))

        c = math.cos(theta)
        s = math.sin(theta)

        app = A[p][p]
        aqq = A[q][q]
        apq = A[p][q]

        A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq
        A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq

        A[p][q] = 0.0
        A[q][p] = 0.0

        for i in range(n):
            if i != p and i != q:
                aip = A[i][p]
                aiq = A[i][q]

                A[i][p] = c*aip - s*aiq
                A[p][i] = A[i][p]

                A[i][q] = s*aip + c*aiq
                A[q][i] = A[i][q]

        for i in range(n):
            vip = V[i][p]
            viq = V[i][q]

            V[i][p] = c*vip - s*viq
            V[i][q] = s*vip + c*viq

    eigenvalues = []
    for i in range(n):
        eigenvalues.append(A[i][i])

    return eigenvalues, V


def sort_eigen(eigenvalues, eigenvectors):

    n = len(eigenvalues)

    for i in range(n):
        max_index = i

        for j in range(i + 1, n):
            if eigenvalues[j] > eigenvalues[max_index]:
                max_index = j

        eigenvalues[i], eigenvalues[max_index] = eigenvalues[max_index], eigenvalues[i]

        for r in range(n):
            eigenvectors[r][i], eigenvectors[r][max_index] = (
                eigenvectors[r][max_index],
                eigenvectors[r][i],
            )


n = int(input("Enter order of matrix N: "))

print("Enter Matrix Elements:")

A = []

for i in range(n):
    row = list(map(float, input().split()))
    A.append(row)

k = int(input("Enter value of K: "))

eigenvalues, eigenvectors = jacobi(A)

sort_eigen(eigenvalues, eigenvectors)

print("\nEigen Values:")

for i in range(n):
    print(f"{eigenvalues[i]:.6f}")

print("\nEigen Vectors:")

for i in range(n):
    print("Eigen Vector", i + 1)
    for j in range(n):
        print(f"{eigenvectors[j][i]:.6f}", end=" ")
    print()

print("\nTop", k, "Principal Components")

for i in range(k):
    print("\nPrincipal Component", i + 1)
    print("Eigen Value =", round(eigenvalues[i], 6))
    print("Eigen Vector:")

    for j in range(n):
        print(round(eigenvectors[j][i], 6), end=" ")
    print()