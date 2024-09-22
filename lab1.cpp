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

vector<vector<T>> MatrixMult(const vector<vector<T>>& A, const vector<vector<T>>& B)
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
    return C;
}

vector<vector<T>> Transpose(const vector<vector<T>>& A) {
    vector<vector<T>> A1 = A;
    int n = A.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A1[i][j] = A[j][i];
        }
    }
    return A1;
}

T vectorNorm1(const vector<T>& b)
{
    int n = b.size();
    T norm = 0;

    for (int i = 0; i < n; i++)
        norm += abs(b[i]);

    return norm;
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

vector<T> QRMethod(vector<vector<T>>& A, vector<T>& b)
{
    int n = A.size();
    vector<T> x(n);
    vector<vector<T>> A1 = A;
    vector<vector<T>> P(n, vector<T>(n));
    vector<vector<T>> R(n, vector<T>(n));
    // генерация единичной матрицы
    for (int i = 0; i < n; i++)
        P[i][i] = 1;

    for (int i = 0; i < n; i++)
    {
        int mx = i;
        for (int j = i; j < n; j++)
        {
            if (abs(A1[j][i]) > abs(A1[mx][i]))
                mx = j;
        }
        if (mx != i)
        {
            swap(A1[mx], A1[i]);
            swap(P[mx], P[i]);
        }
        if (abs(A1[i][i]) < 1e-9)
            system("pause");

        for (int j = i + 1; j < n; j++)
        {
            T ckj = A1[i][i] / sqrt(A1[i][i] * A1[i][i] + A1[j][i] * A1[j][i]);
            T sjk = A1[j][i] / sqrt(A1[i][i] * A1[i][i] + A1[j][i] * A1[j][i]);

            for (int k = 0; k < n; k++)
            {
                T temp = ckj * A1[i][k] + sjk * A1[j][k];
                A1[j][k] = -sjk * A1[i][k] + ckj * A1[j][k];
                A1[i][k] = temp;
            }

            for (int k = 0; k < n; k++)
            {
                T temp = ckj * P[i][k] + sjk * P[j][k];
                P[j][k] = -sjk * P[i][k] + ckj * P[j][k];
                P[i][k] = temp;
            }
        }
    }
    
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (j >= i)
                R[i][j] = A1[i][j];
        }
    }

    vector<vector<T>> Q = Transpose(P);
    cout << endl << "Матрица Q:" << endl;
    printMatrix(Q);
    cout << endl << "Матрица R:" << endl;
    printMatrix(R);
    cout << endl << "Матрица A (проверка):" << endl;
    printMatrix(MatrixMult(Q, R));
    cout << endl;

    vector<T> b1(n, 0);
    for (int i = 0; i < n; i++) {
        T temp = 0;
        for (int j = 0; j < n; j++) { temp += Q[j][i] * b[j]; }
        b1[i] = temp;
    }

    for (int i = n - 1; i >= 0; i--) {
        x[i] = b1[i];
        for (int j = i + 1; j < n; j++) { x[i] -= R[i][j] * x[j]; }
        x[i] /= R[i][i];
    }
    return x;
}

void ResidualVectorNorm(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x) // норма вектора невязки
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

    T norm1 = vectorNorm1(residualVec);
    T norm2 = vectorNormInf(residualVec);
    cout << endl << "Норма вектора невязки ||b - b1||" << endl;
    cout << "для октаэдрической нормы: " << norm1 << endl;
    cout << "для кубической нормы: " << norm2 << endl;
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

// Число обусловленности для различных матричных норм
T cond1(const vector<vector<T>>& A)
{
    vector<vector<T>> Ainv = InvLU(A);
    return matrixNorm1(Ainv) * matrixNorm1(A);
}

T condInf(const vector<vector<T>>& A)
{
    vector<vector<T>> Ainv = InvLU(A);
    return matrixNormInf(Ainv) * matrixNormInf(A);
}

void condEstimation(vector<vector<T>>& A, vector<T>& b, const vector<T>& disturb)
{
    int n = b.size();
    vector<T> b1(n);
    for (int i = 0; i < n; i++)
        b1[i] = b[i] + disturb[i];
    
    vector<T> x(n), x1(n);
    x = GaussianMethod(A, b);
    x1 = GaussianMethod(A, b1);

    cout << "Решение возмущенной системы:" << endl;
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << x1[i] << endl;

    T b_delta1 = vectorNorm1(disturb) / vectorNorm1(b);
    T b_deltaInf = vectorNormInf(disturb) / vectorNormInf(b);

    vector<T> dx(n);
    for (int i = 0; i < n; i++)
        dx[i] = abs(x1[i] - x[i]);

    T x_delta1 = vectorNorm1(dx) / vectorNorm1(x);
    T x_deltaInf = vectorNormInf(dx) / vectorNormInf(x);

    T condEst1 = x_delta1 / b_delta1;
    T condEstInf = x_deltaInf / b_deltaInf;

    cout << endl << condEst1 << endl;
    cout << condEstInf << endl;
}

void condEstimationLower(vector<vector<T>>& A, vector<T>& b, int disturbCount = 5)
{
    int n = b.size();
    T lower1 = 1e+9;
    T lowerInf = 1e+9;

    vector<T> x(n);
    x = GaussianMethod(A, b);

    vector<T> disturb = { -0.01, 0.01 };
    for (int i = 0; i < disturbCount; i++)
    {
        vector<T> b1(n);
        for (int j = 0; j < n; j++)
            b1[j] = b[j] + disturb[rand() % 2];

        // for (int j = 0; j < n; j++)
        //    cout << "b" << j + 1 << " = " << b1[j] << endl;

        vector<T> x1(n);
        x1 = GaussianMethod(A, b1);

        // cout << "Решение возмущенной системы:" << endl;
        // for (int k = 0; k < n; k++)
        //    cout << "x" << k + 1 << " = " << x1[k] << endl;

        vector<T> db(n);
        for (int j = 0; j < n; j++)
            db[j] = abs(b1[j] - b[j]);

        vector<T> dx(n);
        for (int j = 0; j < n; j++)
            dx[j] = abs(x1[j] - x[j]);

        T b_delta1 = vectorNorm1(db) / vectorNorm1(b);
        T b_deltaInf = vectorNormInf(db) / vectorNormInf(b);

        T x_delta1 = vectorNorm1(dx) / vectorNorm1(x);
        T x_deltaInf = vectorNormInf(dx) / vectorNormInf(x);

        T condEst1 = x_delta1 / b_delta1;
        T condEstInf = x_deltaInf / b_deltaInf;

        // cout << endl << condEst1 << endl;
        // cout << condEstInf << endl;

        if (condEst1 > 0)
            lower1 = min(lower1, condEst1);
        if (condEstInf > 0)
            lowerInf = min(lowerInf, condEstInf);
    }

    cout << endl << "cond1 A >= " << lower1 << endl;
    cout << "condInf A >= " << lowerInf << endl;
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

    ResidualVectorNorm(A, b, solution1);

    cout << endl << "Решение СЛАУ методом QR-разложения:" << endl;
    vector<T> solution2 = QRMethod(A, b);
    for (int i = 0; i < n; i++)
        cout << "x" << i + 1 << " = " << solution2[i] << endl;

    ResidualVectorNorm(A, b, solution2);

    vector<vector<T>> Ainv = InvLU(A);
    cout << endl << "Обратная матрица для матрицы системы A:" << endl;
    printMatrix(Ainv);

    cout << endl << "Число обусловленности матрицы A при использовании" << endl;
    cout << "октаэдрической нормы: " << cond1(A) << endl;
    cout << "кубической нормы: " << condInf(A) << endl;

    // vector<T> disturb = { 0.01, 0.01, 0.01, 0.01 }; // возмущение

    // condEstimation(A, b, disturb);

    cout << endl << "Оценка числа обусловленности:";
    condEstimationLower(A, b);

    cout << endl << "Результат умножения A^(-1) на A:" << endl;
    vector<vector<T>> M = MatrixMult(A, Ainv);
    printMatrix(M);
}
