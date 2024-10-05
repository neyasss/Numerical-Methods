#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include "Header.h"

using namespace std;

void readSLAE(const string& file, vector<vector<T>>& A, vector<T>& b)
{
    ifstream fin(file);
    if (fin.is_open())
    {
        int n;
        fin >> n;
        A.resize(n, vector<T>(n));
        b.resize(n);
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                fin >> A[i][j];
            }
            fin >> b[i];
        }
    }
    fin.close();
}

void printSLAE(const vector<vector<T>>& A, const vector<T>& b, int n) // вывод СЛАУ
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << A[i][j] << "x" << j + 1;
            if (j < n - 1)
                cout << " + ";
        }
        cout << " = " << b[i] << endl;
    }
}

void printMatrix(const vector<vector<T>>& A, int n) // вывод матрицы
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B, int n)
{
    vector<vector<T>> C(n, vector<T>(n));
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            C[i][j] = 0;
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
        }
    }
    return C;
}

vector<vector<T>> Transpose(const vector<vector<T>>& A, int n)
{
    vector<vector<T>> A1 = A;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A1[i][j] = A[j][i];
        }
    }
    return A1;
}

T vectorNorm1(const vector<T>& b, int n)
{
    T norm = 0;

    for (int i = 0; i < n; i++)
        norm += abs(b[i]);

    return norm;
}

T vectorNormInf(const vector<T>& b, int n)
{
    T norm = -1;

    for (int i = 0; i < n; i++)
    {
        if (abs(b[i]) > norm)
            norm = abs(b[i]);
    }

    return norm;
}

T ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, int n, int norm) // норма вектора невязки
{
    vector<T> residualVec(n);
    T result = 0;
    for (int i = 0; i < n; i++)
    {
        residualVec[i] = b[i];
        for (int j = 0; j < n; j++)
        {
            residualVec[i] -= A[i][j] * x[j]; // вычисляем компоненты вектора невязки
        }
    }

    if (norm == 1)
    {
        T norm1 = vectorNorm1(residualVec, n);
        return norm1;
    }
    if (norm == 0)
    {
        T normInf = vectorNormInf(residualVec, n);
        return normInf;
    }
    else
        return 0;
}

vector<vector<T>> InvLU(const vector<vector<T>>& A, int n) // нахождение обратной матрицы с помощью LU-разложения
{
    vector<vector<T>> LU = A; // L и U хранятся как одна матрица
    vector<vector<T>> Ainv(n, vector<T>(n));

    for (int i = 0; i < n; i++)
    {
        if (LU[i][i] == 0)
        {
            for (int j = i + 1; j < n; j++)
            {
                if (LU[j][i] != 0)
                    swap(LU[i], LU[j]);
            }
        }
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            for (int k = 0; k < i; k++)
                LU[i][j] -= LU[i][k] * LU[k][j]; // элементы матрицы L (нижнетреугольная с единицами на главной диагонали)
        }
        for (int j = i + 1; j < n; j++)
        {
            for (int k = 0; k < i; k++)
                LU[j][i] -= LU[j][k] * LU[k][i]; // элементы матрицы U (верхнетреугольная)
            LU[j][i] /= LU[i][i];
        }
    }

    // printMatrix(LU, n);

    for (int i = 0; i < n; i++)
        Ainv[i][i] = 1;

    for (int col = 0; col < n; col++)
    {
        vector<T> y(n), x(n);
        for (int i = 0; i < n; i++)     // Решаем систему Ly = b (b - столбцы из единичной матрицы)
        {
            y[i] = Ainv[i][col];
            for (int j = 0; j < i; j++)
                y[i] -= LU[i][j] * y[j];
        }

        for (int i = n - 1; i >= 0; i--)     // Решаем систему Ux = y
        {
            x[i] = y[i];
            for (int j = i + 1; j < n; j++)
                x[i] -= LU[i][j] * x[j];
            x[i] /= LU[i][i];
        }

        for (int i = 0; i < n; i++)
            Ainv[i][col] = x[i];
    }

    return Ainv;
}


T matrixNorm1(const vector<vector<T>>& A, int n) // октаэдрическая норма
{
    vector<T> sum(n); // хранятся суммы модулей элементов всех столбцов
    T maxSum = -1;

    for (int j = 0; j < n; j++)
    {
        sum[j] = 0;
        for (int i = 0; i < n; i++)
            sum[j] += abs(A[i][j]);
        if (sum[j] > maxSum)
            maxSum = sum[j];
    }

    return maxSum;
}

T matrixNormInf(const vector<vector<T>>& A, int n) // кубическая норма
{
    vector<T> sum(n); // хранятся суммы модулей элементов всех строк
    T maxSum = -1;

    for (int i = 0; i < n; i++)
    {
        sum[i] = 0;
        for (int j = 0; j < n; j++)
            sum[i] += abs(A[i][j]);
        if (sum[i] > maxSum)
            maxSum = sum[i];
    }

    return maxSum;
}

// Число обусловленности для различных матричных норм
T cond1(const vector<vector<T>>& A, int n)
{
    vector<vector<T>> Ainv = InvLU(A, n);
    return matrixNorm1(Ainv, n) * matrixNorm1(A, n);
}

T condInf(const vector<vector<T>>& A, int n)
{
    vector<vector<T>> Ainv = InvLU(A, n);
    return matrixNormInf(Ainv, n) * matrixNormInf(A, n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////
// 
// лр2

// переопределение некоторых операторов для удобства использования в итерационных методах
vector<vector<T>> operator*(const vector<vector<T>>& A, const T& num) // умножение матрицы на число
{
    int n = A.size();
    vector<vector<T>> B = A;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            B[i][j] *= num;
    }
    return B;
}

vector<T> operator*(const vector<vector<T>>& A, const vector<T>& x) // умножение матрицы на вектор
{
    int n = A.size();
    vector<T> y(n);
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            y[i] += A[i][j] * x[j];
    }
    return y;
}

vector<vector<T>> operator+(const vector<vector<T>>& A, const vector<vector<T>>& B) // сложение матриц
{
    int n = A.size();
    vector<vector<T>> C = A;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            C[i][j] += B[i][j];
    }
    return C;
}

vector<vector<T>> operator-(const vector<vector<T>>& A, const vector<vector<T>>& B) // вычитание матриц
{
    int n = A.size();
    vector<vector<T>> C = A;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            C[i][j] -= B[i][j];
    }
    return C;
}

vector<T> operator+(const vector<T>& x, const vector<T>& y) // сложение векторов
{
    int n = x.size();
    vector<T> v = x;
    for (int i = 0; i < n; i++)
        v[i] += y[i];
    return v;
}

vector<T> operator-(const vector<T>& x, const vector<T>& y) // вычитание векторов
{
    int n = x.size();
    vector<T> v = x;
    for (int i = 0; i < n; i++)
        v[i] -= y[i];
    return v;
}

// Метод простых итераций
Params SimpleIterationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm)
{
    Params params;
    vector<T> y(n), xk(n), xnext(n);
    int maxIter = 1000;
    xk = x0; // x0 - начальное приближение, xk - на k-м шаге итерации
    xnext = xk; // k+1 итерация
    for (int i = 0; i < n; i++)
        y[i] = tau * b[i];
    params.y = y;

    vector<vector<T>> C(n, vector<T>(n)), E(n, vector<T>(n));
    for (int i = 0; i < n; i++)
        E[i][i] = 1;

    C = E - A * tau;
    printMatrix(C, n);
    params.C = C;
    params.normC1 = matrixNorm1(C, n);
    params.normCInf = matrixNormInf(C, n);

    for (int iter = 0; iter < maxIter; iter++)
    {
        xnext = C * xk + y;

        // Критерии останова итерационного процесса
        if (norm == 1)
        {
            if (ResidualVectorNorm(A, b, xnext, n, norm) < eps || vectorNorm1(xnext - xk, n) < eps
                || vectorNorm1(xnext - xk, n) <= (1 - matrixNorm1(C, n)) / matrixNorm1(C, n) * eps)
            {
                params.x = xnext;
                params.iterCount = iter + 1;
                return params;
            }
        }

        if (norm == 0)
        {
            if (ResidualVectorNorm(A, b, xnext, n, norm) < eps || vectorNormInf(xnext - xk, n) < eps
                || vectorNormInf(xnext - xk, n) <= (1 - matrixNormInf(C, n)) / matrixNormInf(C, n) * eps)
            {
                params.x = xnext;
                params.iterCount = iter + 1;
                return params;
            }
        }
        xk = xnext;
    }
    params.x = xk;
    params.iterCount = maxIter;
    return params;
}

void LDU(vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U, int n) // представление матрицы A = L + D + U
{
    for (int i = 0; i < n; i++)
    {
        D[i][i] = A[i][i];
        for (int j = 0; j < n; j++)
        {
            if (i < j)
                U[i][j] = A[i][j];
            if (i > j)
                L[i][j] = A[i][j];
        }
    }
}


Params JacobiMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& eps, int n, int norm)
{
    // С помощью LDU-разложения
    /*
    Params params;
    vector<T> y(n), xk(n), xnext(n);
    xk = x0;
    xnext = xk;
    int maxIter = 100;
    vector<vector<T>> L(n, vector<T>(n)), D(n, vector<T>(n)), U(n, vector<T>(n));
    LDU(A, L, D, U, n);
    vector<vector<T>> C(n, vector<T>(n)), invD(n, vector<T>(n));
    for (int i = 0; i < n; i++)
        invD[i][i] = 1 / D[i][i];

    y = invD * b;
    params.y = y;

    C = MatrixMult(invD, L + U, n) * (-1);
    params.C = C;
    */

    // По формулам (методичка)
    Params params;
    vector<T> y(n), xk(n), xnext(n);
    xk = x0;
    xnext = xk;
    int maxIter = 100;
    vector<vector<T>> C(n, vector<T>(n));
    for (int i = 0; i < n; i++)
    {
        y[i] = b[i] / A[i][i];
        for (int j = 0; j < n; j++)
        {
            if (i != j)
                C[i][j] = -A[i][j] / A[i][i];
        }
    }

    params.y = y;
    params.C = C;

    for (int iter = 0; iter < maxIter; iter++)
    {
        xnext = C * xk + y;

        if (norm == 1)
        {
            if (ResidualVectorNorm(A, b, xnext, n, norm) < eps || vectorNorm1(xnext - xk, n) < eps
                || vectorNorm1(xnext - xk, n) <= (1 - matrixNorm1(C, n)) / matrixNorm1(C, n) * eps)
            {
                params.x = xnext;
                params.iterCount = iter + 1;
                return params;
            }
        }

        if (norm == 0)
        {
            if (ResidualVectorNorm(A, b, xnext, n, norm) < eps || vectorNormInf(xnext - xk, n) < eps
                || vectorNormInf(xnext - xk, n) <= (1 - matrixNormInf(C, n)) / matrixNormInf(C, n) * eps)
            {
                params.x = xnext;
                params.iterCount = iter + 1;
                return params;
            }
        }
        xk = xnext;
    }
    params.x = xk;
    params.iterCount = maxIter;
    return params;
}

/*
Params SeidelMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm)
{

}

Params RelaxationMethod(vector<vector<T>>& A, vector<T> b, vector<T> x0, const T& tau, const T& eps, int n, int norm)
{

}
*/
