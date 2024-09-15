#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
using namespace std;

// #define T float                          // обычная точность 
#define T double                            // повышенная точность

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

void printSLAE(const vector<vector<T>>& A, const vector<T>& b)
{
    int n = A.size();
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


// Метод Гаусса с частичным выбором главного элемента (по столбцу)
vector<T> GaussianMethod(vector<vector<T>>& A, vector<T>& b)
{
    int n = A.size();
    vector<T> x(n);
    vector<vector<T>> A1 = A;
    vector<T> b1 = b;

    for (int i = 0; i < n; i++) // ищем главный элемент
    {
        T maxElem = abs(A1[i][i]);
        int maxRow = i;
        for (int k = i + 1; k < n; k++)
        {
            if (abs(A1[k][i]) > maxElem)
            {
                maxElem = abs(A1[k][i]);
                maxRow = k;
            }
        }

        if (maxElem < 1e-9) // проверка на нулевые (или малые по абсолютной величине) диагональные элементы
        {
            cout << "Не существует единственного решения СЛАУ" << endl;
            return x;
        }
        
        swap(A1[i], A1[maxRow]); // переставляем текущую строку со строкой с главным элементом
        swap(b1[i], b1[maxRow]);

        for (int j = i + 1; j < n; j++) // прямой ход
        {
            T c_ji = A1[j][i] / A1[i][i];
            for (int k = i; k < n; k++)
                A1[j][k] -= c_ji * A1[i][k];
            b1[j] -= c_ji * b1[i];
        }
    }

    // printSLAE(A1, b1); // полученный верхнетреугольный вид

    for (int i = n - 1; i >= 0; i--) // обратный ход
    {
        x[i] = b1[i];
        for (int j = i + 1; j < n; j++)
            x[i] -= A1[i][j] * x[j];
        x[i] /= A1[i][i];
    }

    return x;
}

T ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x) // норма вектора невязки
{
    int n = A.size();
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

    for (T elem : residualVec)
        result += elem * elem;

    return sqrt(result);
}

vector<vector<T>> InvLU(const vector<vector<T>>& A) // нахождение обратной матрицы с помощью LU-разложения
{
    int n = A.size();
    vector<vector<T>> LU(n, vector<T>(n)); // L и U хранятся как одна матрица
    vector<vector<T>> Ainv(n, vector<T>(n));

    for (int i = 0; i < n; i++)
    {
        for (int j = i; j < n; j++)
        {
            LU[i][j] = A[i][j];
            for (int k = 0; k < i; k++)
                LU[i][j] -= LU[i][k] * LU[k][j]; // элементы матрицы L (нижнетреугольная с единицами на главной диагонали)
        }
        for (int j = i + 1; j < n; j++)
        {
            LU[j][i] = A[j][i];
            for (int k = 0; k < i; k++)
                LU[j][i] -= LU[j][k] * LU[k][i]; // элементы матрицы U (верхнетреугольная)
            LU[j][i] /= LU[i][i];
        }
    }

    // пока работает некорректно; разложение LU правильное
    vector<T> y(n), x(n);
    for (int i = 0; i < n; i++)
    {
        y[i] = 1;
        for (int j = 0; j < i; j++)     // Решаем систему Ly = b (b = E, E - единичная матрица)
            y[i] -= LU[i][j] * y[j];

        for (int j = n - 1; j >= 0; j--)     // Решаем систему Ux = y
        {
            Ainv[j][i] = y[j];
            for (int k = j + 1; k < n; k++)
                Ainv[j][i] -= LU[j][k] * Ainv[k][i];
            Ainv[j][i] /= LU[j][j];
        }
    }

    return Ainv;
}

int main()
{
    setlocale(LC_ALL, "Russian");
    vector<vector<T>> A;
    vector<T> b;
    readSLAE("test.txt", A, b);
    int n = A.size();
    // vector<int> order(n);
    // for (int i = 0; i < n; i++)
    //     order[i] = i;

    cout << "Исходная СЛАУ:" << endl;
    printSLAE(A, b);
    cout << endl;

    cout << "Решение СЛАУ методом Гаусса:" << endl;
    vector<T> solution1 = GaussianMethod(A, b);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << solution1[i] << endl;

    cout << "Норма вектора невязки ||b - b1||: ";
    T norm = ResidualVectorNorm(A, b, solution1);
    cout << norm << endl;

    vector<vector<T>> Ainv = InvLU(A);
}
