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

void printSLAE(const vector<vector<T>>& A, const vector<T>& b) // вывод СЛАУ
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

void printMatrix(const vector<vector<T>>& A) // вывод матрицы
{
    int n = A.size();
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            cout << A[i][j] << " ";
        cout << endl;
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
            system("pause");
        }
        
        swap(A1[i], A1[maxRow]); // переставляем текущую строку матрицы со строкой с главным элементом
        swap(b1[i], b1[maxRow]); // аналогично для правой части

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

    // printMatrix(LU);

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

T matrixNorm1(const vector<vector<T>>& A) // октаэдрическая норма
{
    int n = A.size();
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

T matrixNorm2(const vector<vector<T>>& A) // шаровая норма
{
    int n = A.size();
    T sum = 0;

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            sum += A[i][j] * A[i][j];
    }

    return sqrt(sum);
}

T matrixNormInf(const vector<vector<T>>& A) // кубическая норма
{
    int n = A.size();
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

T vectorNorm1(const vector<T>& b)
{
    int n = b.size();
    T norm = 0;

    for (int i = 0; i < n; i++)
        norm += abs(b[i]);

    return norm;
}

T vectorNorm2(const vector<T>& b)
{
    int n = b.size();
    T result = 0;

    for (int i = 0; i < n; i++)
        result += b[i] * b[i];

    return sqrt(result);
}

T vectorNormInf(const vector<T>& b)
{
    int n = b.size();
    T norm = -1;

    for (int i = 0; i < n; i++)
    {
        if (abs(b[i]) > norm)
            norm = abs(b[i]);
    }

    return norm;
}

// Число обусловленности для различных матричных норм
T cond1(const vector<vector<T>>& A) 
{
    vector<vector<T>> Ainv = InvLU(A);
    return matrixNorm1(Ainv) * matrixNorm1(A);
}

T cond2(const vector<vector<T>>& A)
{
    vector<vector<T>> Ainv = InvLU(A);
    return matrixNorm2(Ainv) * matrixNorm2(A);
}

T condInf(const vector<vector<T>>& A)
{
    vector<vector<T>> Ainv = InvLU(A);
    return matrixNormInf(Ainv) * matrixNormInf(A);
}

T delta(const vector<T>& x, const vector<T>& x1) // x1 - решение возмущенной системы
{
    int n = x.size();
    vector<T> dx(n);
    T delta = 0;
    for (int i = 0; i < n; i++)
        dx[i] = abs(x1[i] - x[i]);
    delta = vectorNorm1(dx) / vectorNorm1(x);
    return delta;
}

T condEstimation(const vector<T>& x, const vector<T>& x1, const vector<T>& b, const vector<T>& b1)
{
    return delta(x, x1) / delta(b, b1);
}

void MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B)
{
    int n = A.size();
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
    
    printMatrix(C);
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

    cout << endl << "Норма вектора невязки ||b - b1||: ";
    T norm = ResidualVectorNorm(A, b, solution1);
    cout << norm << endl;

    vector<vector<T>> Ainv = InvLU(A);
    cout << endl << "Обратная матрица для матрицы системы A:" << endl;
    printMatrix(Ainv);

    cout << endl << "Число обусловленности матрицы A при использовании" << endl;
    cout << "октаэдрической нормы: " << cond1(A) << endl;
    // cout << "шаровой нормы: " << cond2(A) << endl;
    cout << "кубической нормы: " << condInf(A) << endl;

    cout << endl << "Внесем возмущение 0.01 в систему:" << endl;
    vector<T> b1(n);
    for (int i = 0; i < n; i++)
        b1[i] = b[i] + 0.01;
    printSLAE(A, b1);

    cout << endl << "Решение возмущенной системы методом Гаусса:" << endl;
    vector<T> solution2 = GaussianMethod(A, b1);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << solution2[i] << endl;

    cout << endl << "Результат умножения A^(-1) на A:" << endl;
    MatrixMult(A, Ainv);

    cout << endl << "Оценка числа обусловленности:" << endl;
    cout << condEstimation(solution1, solution2, b, b1) << endl;
}
