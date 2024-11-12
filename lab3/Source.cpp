#include "Header.h"

using namespace std;

/* *** Начальные функции для испорта/экспорта данных *** */

/* Функция импорта матрицы СЛАУ из текстового файла*/
template <typename T>
vector<vector<T>> importSLAU(const string& filename) {
    vector<vector<T>> matrix;
    vector<T> vec;
    ifstream file(filename);

    if (!file.is_open()) {
        printf("Error: not open file \n");
        exit(1);
    }

    int size;
    file >> size;

    matrix.resize(size, vector<T>(size + 1));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size + 1; ++j) {
            T value;
            if (file >> value) {
                matrix[i][j] = value;
            }
        }
    }

    file.close();
    return matrix;
};

/* Функция импорта матрицы из текстового файла*/
template <typename T>
vector<vector<T>> importMatrix(const string& filename) {
    vector<vector<T>> matrix;
    ifstream file(filename);

    if (!file.is_open()) {
        printf("Error: not open file \n");
        exit(1);
    }

    int size;
    file >> size;

    matrix.resize(size, vector<T>(size));

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            T value;
            if (file >> value) {
                matrix[i][j] = value;
            }
        }
    }

    file.close();
    return matrix;
};


/* Функция вывода матрицы на экран */
template <typename T>
void print(const vector<vector<T>>& matrix) {
    for (vector<T> row : matrix) {
        for (T value : row) {
            cout << value << ' ';
        }
        cout << '\n';
    }
    cout << endl;
}

/* Функция вывода вектора на экран */
template <typename T>
void print(const vector<T>& vec) {
    for (T value : vec) {
        cout << value << ' ';
    }
    cout << endl;
}

/* Функция вывода обрезанного вектора на экран */
template <typename T>
void print_short(const vector<T>& vec, const int& n) {

    for (int i = 0; i < n; ++i) {
        cout << vec[i] << ' ';
    }
    cout << "..." << endl;
}

/* Функция, которая красиво выводит вектор*/
template<typename T>
void print_vec(const vector<T>& vec) {
    cout << "(" << vec[0];
    for (int i = 1; i < vec.size(); i++) {
        cout << ", " << vec[i];
    }
    cout << ")" << endl;
}


/* Функция для получения матрицы из СЛАУ */
template <typename T>
vector<vector<T>> SLAU_to_matrix(const vector<vector<T>>& SLAU) {
    vector<vector<T>> matrix;
    matrix.resize(SLAU.size(), vector<T>(SLAU.size()));

    for (int i = 0; i < SLAU.size(); i++) {
        for (int j = 0; j < SLAU.size(); j++) {
            matrix[i][j] = SLAU[i][j];
        }
    }
    return matrix;
}


/* Функция для получения вектора из СЛАУ */
template <typename T>
vector<T> SLAU_to_vec(const vector<vector<T>>& SLAU) {
    int s = SLAU.size();
    vector<T> vec(s);

    for (int i = 0; i < SLAU.size(); i++) {
        vec[i] = SLAU[i][s];
    }
    return vec;
}


/* *** Функции математики векторов *** */

/* Функция для сложения векторов */
template <typename T>
vector<T> operator+(const vector<T>& vec1, const vector<T>& vec2) {
    vector<T> pert_vec = vec1;
    for (int i = 0; i < vec1.size(); i++) {
        pert_vec[i] += vec2[i];
    }
    return pert_vec;
}


/* Функция вычитания векторов */
template <typename T>
vector<T> operator-(const vector<T>& a, const vector<T>& b) {
    // Проверка на возможность умножения
    if (a.size() != b.size()) {
        cout << "Error: size a != size b in substraction vectors." << endl;
        exit(1);
    }
    // Создание результирующего вектора
    vector<T> result(a.size(), 0);

    // Умножение матрицы на вектор
    for (int i = 0; i < a.size(); ++i) {
        result[i] += a[i] - b[i];
    }
    return result;

}

/* Операция почленного умножения векторов */
template <typename T>
vector<T> operator*(const vector<T>& vec1, const vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        cout << "Error: vector1 size != vector2 size in operator*." << endl;
        exit(1);
    }
    vector<T> result(vec1.size(), 0);
    for (int i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] * vec2[i];
    }
    return result;
}

/* Операция умножения вектора на число */
template <typename T>
vector<T> operator*(const T& c, const vector<T>& vec) {
    vector<T> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * c;
    }
    return result;
}

template <typename T>
vector<T> operator*(const vector<T>& vec, const T& c) {
    vector<T> result(vec.size(), 0);
    for (int i = 0; i < vec.size(); i++) {
        result[i] = vec[i] * c;
    }
    return result;
}


/* Операция почленного деления векторов */
template <typename T>
vector<T> operator/(const vector<T>& vec1, const vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        cout << "Error: vector1 size != vector2 size in operator*." << endl;
        exit(1);
    }
    vector<T> result(vec1.size(), 0);
    for (int i = 0; i < vec1.size(); i++) {
        result[i] = vec1[i] / vec2[i];
    }
    return result;
}

/* Функция для скалярного умножения векторов */
template <typename T>
T dot(const vector<T>& vec1, const vector<T>& vec2) {
    if (vec1.size() != vec2.size()) {
        cout << "Error: vector1 size != vector2 size in operator*." << endl;
        exit(1);
    }
    T result{ 0 };
    for (int i = 0; i < vec1.size(); i++) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

/* Функция для нормы вектора */
template <typename T>
T norm(const vector<T>& vec, const int& p) {
    if (vec.empty()) {
        cerr << "Error: Empty vector in norm() \n";
        exit(1);
    }

    T result = 0.0;

    // Вычисление нормы
    if (p == 0) {
        // Норма oo
        for (const auto& element : vec) {
            T absElement = abs(element);
            if (absElement > result) {
                result = absElement;
            }
        }
    }
    else {
        // Общий случай для норм L1, L2 и т.д.
        for (const auto& element : vec) {
            result += pow(abs(element), p);
        }

        result = pow(result, 1.0 / p);
    }

    return result;
}

/* Функция, которая возращает матрицу комбинаций жлементов вектора */
template<typename T>
vector<vector<T>> generateCombinations(const vector<T>& vec) {
    int n = vec.size();

    // Вектор для хранения всех комбинаций
    vector<vector<T>> combinations;

    // Внешний цикл по всем возможным комбинациям
    for (int i = 0; i < (1 << n); ++i) {
        vector<T> current(n);

        // Внутренний цикл для каждой позиции вектора
        for (int j = 0; j < n; ++j) {
            current[j] = (i & (1 << j)) ? vec[j] : -vec[j];
        }

        // Добавить текущую комбинацию в вектор
        combinations.push_back(current);
    }

    return combinations;
}

/* Функция, возвращает вектор модулей */
template<typename T>
vector<T> vec_abs(const vector<T> vec) {
    for (int i = 0; i < vec.size(); i++) {
        vec[i] = fabs(vec[i]);
    }
    return vec;
}

/* Функция, возращающая сумму элементов вектора */
template<typename T>
T sum(const vector<T>& vec) {
    T sum = 0;
    for (int i = 0; i < vec.size(); i++) {
        sum += vec[i];
    }
    return sum;
}

/* *** Функции математики матриц *** */

/* Операция для умножения матрицы на число */
template <typename T>
vector<vector<T>> operator*(const vector<vector<T>>& A, const T& scalar) {
    // Создание результирующей матрицы с теми же размерами
    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));

    // Умножение каждого элемента матрицы на число
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] * scalar;
        }
    }

    return result;
}

/* Операция для умножения  числа на матрицу */
template <typename T>
vector<vector<T>> operator*(const T& scalar, const vector<vector<T>>& A) {
    // Создание результирующей матрицы с теми же размерами
    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));

    // Умножение каждого элемента матрицы на число
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] * scalar;
        }
    }

    return result;
}

/* Операция поэлементного сложения матриц */
template <typename T>
vector<vector<T>> operator+(const vector<vector<T>>& A, const vector<vector<T>>& B) {
    // Проверка на совпадение размеров матриц
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        cout << "Error: size A != size B in addition matrix." << endl;
        exit(1);
    }

    // Создание результирующей матрицы с теми же размерами
    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));

    // Поэлементное сложение
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }

    return result;
}

/* Операция поэлементного вычитания матриц */
template <typename T>
vector<vector<T>> operator-(const vector<vector<T>>& A, const vector<vector<T>>& B) {
    // Проверка на совпадение размеров матриц
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        cout << "Error: size A != size B in substraction matrix." << endl;
        exit(1);
    }

    // Создание результирующей матрицы с теми же размерами
    vector<vector<T>> result(A.size(), vector<T>(A[0].size(), 0));

    // Поэлементное сложение
    for (size_t i = 0; i < A.size(); ++i) {
        for (size_t j = 0; j < A[0].size(); ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

/* Операция умножения матрицы на вектор */
template <typename T>
vector<T> operator*(const vector<vector<T>>& matrix, const vector<T>& vec) {
    // Проверка на возможность умножения
    if (matrix[0].size() != vec.size()) {
        cout << "Error: size A != size b in multiply Matrix By Vector." << endl;
        exit(1);
    }
    // Создание результирующего вектора
    vector<T> result(matrix.size(), 0);

    // Умножение матрицы на вектор
    for (int i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[0].size(); ++j) {
            result[i] += matrix[i][j] * vec[j];
        }
    }
    return result;
}


/* Матричное умножение */
template <typename T>
vector<vector<T>> operator*(const vector<vector<T>>& A, const vector<vector<T>>& B) {
    int m = A.size();    // Количество строк в матрице A
    int n = A[0].size(); // Количество столбцов в матрице A
    int p = B[0].size(); // Количество столбцов в матрице B

    if (n != B.size()) {
        printf("Error: impossible multiply matrix");
        exit(1);
    }

    vector<vector<T>> result(m, vector<T>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Определение оператора отрицания для матрицы
template <typename T>
vector<vector<T>> operator-(const vector<vector<T>>& matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();
    vector<vector<T>> result(rows, vector<T>(cols, 0));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = -matrix[i][j];
        }
    }
    return result;
}

/* Функция для поэлементного умножения матриц */
template <typename T>
vector<vector<T>> Multyply(const vector<vector<T>>& A, const vector<vector<T>>& B) {
    int m = A.size();    // Количество строк в матрице A
    int n = A[0].size(); // Количество столбцов в матрице A
    int p = B[0].size(); // Количество столбцов в матрице B

    if (n != B.size()) {
        printf("Error: impossible multiply matrix");
        exit(1);
    }

    vector<vector<T>> result(m, vector<T>(p, 0.0));

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            result[i][j] = A[i][j] * B[i][j];
        }
    }
    return result;
}


/* Функция округления чисел в матрицах */
template <typename T>
vector<vector<T>> Matrix_round(const vector<vector<T>>& A, const double& eps) {
    vector<vector<T>> roundA = A;
    int size = A.size();

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            roundA[i][j] = (round(A[i][j]) >= 0) ? round(abs(A[i][j]) * (1 / eps)) / (1 / eps) : -1 * round(abs(A[i][j]) * (1 / eps)) / (1 / eps);
        }
    }
    return roundA;
}


/* Функция для вычисления нормы матрицы */
template <typename T>
T norm(const vector<vector<T>>& matrix, const int& p) {
    // Проверка на пустую матрицу
    if (matrix.empty() || matrix[0].empty()) {
        cout << "Error: Empty matrix in norm()\n";
        exit(1);
    }

    int rows = matrix.size();
    int cols = matrix[0].size();

    T result = 0.0;

    // Вычисление нормы матрицы
    if (p == 0) {
        // Норма матрицы Чебышева (максимальное значение по модулю в строке)
        for (int i = 0; i < rows; ++i) {
            T rowSum = 0.0;
            for (int j = 0; j < cols; ++j) {
                rowSum += abs(matrix[i][j]);
            }
            if (rowSum > result) {
                result = rowSum;
            }
        }
    }
    else {
        // Общий случай для норм матрицы (Фробениуса и др.)
        for (int j = 0; j < cols; ++j) {
            T colSum = 0.0;
            for (T i = 0; i < rows; ++i) {
                colSum += pow(abs(matrix[i][j]), p);
            }
            result += pow(colSum, 1.0 / p);
        }

        result = pow(result, 1.0 / p);
    }
    return result;
}


/* Функция поворота матрицы вправо */
template <typename T>
vector<vector<T>> RotateRight(const vector<vector<T>>& A) {

    vector<vector<T>> A_rotate(A.size(), vector<T>(A.size(), 0));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            A_rotate[A.size() - 1 - j][i] = A[i][j];
        }
    }

    return A_rotate;

}

/* Функция поворота матрицы влево */
template <typename T>
vector<vector<T>> RotateLeft(const vector<vector<T>>& A) {

    vector<vector<T>> A_rotate(A.size(), vector<T>(A.size(), 0));

    for (int i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A.size(); ++j) {
            A_rotate[j][A.size() - 1 - i] = A[i][j];
        }
    }

    return A_rotate;
}

// Функция для создания единичной матрицы размера n x n
template <typename T>
vector<vector<T>> create_identity_matrix(const int& n) {
    vector<vector<T>> identity(n, vector<T>(n, 0));
    for (int i = 0; i < n; i++) {
        identity[i][i] = 1;
    }
    return identity;
}

template <typename T>
vector<vector<T>> E(const int& n) {
    vector<vector<T>> identity(n, vector<T>(n, 0));
    for (int i = 0; i < n; i++) {
        identity[i][i] = 1;
    }
    return identity;
}

// Функция для обратной матрицы с проверкой на вырожденность
template <typename T>
vector<vector<T>> inverseMatrix(const vector<vector<T>>& A, const T& eps) {
    vector<vector<T>> E = create_identity_matrix<T>(A.size());
    vector<vector<T>> E_rotate = RotateLeft(E);
    vector<T> e(A.size());
    vector<vector<T>> X(A.size(), vector<T>(A.size(), 0));


    for (int i = 0; i < A.size(); i++) {
        e = E_rotate[i];
        X[i] = method_Gaussa(A, e, eps);

    }
    vector<vector<T>> A_inv = RotateLeft(X);
    return A_inv;
}

// Функция для обратной матрицы с проверкой на вырожденность с максимальной точностью
template <typename T>
vector<vector<T>> inverseMatrix(const vector<vector<T>>& A) {
    T eps = numeric_limits<T>::epsilon();
    return inverseMatrix(A, eps);
}

/* Функция для вычисления числа обусловленности матрицы c нормой 1*/
template <typename T>
T cond_1(const vector<vector<T>>& matrix) {
    T n_1 = norm_1(matrix);
    if (n_1 == 0) {
        printf("Error: Det(A) = 0  =>  cond_1(A) = oo");
        return numeric_limits<T>::infinity();
    }
    vector<vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_1(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}

/* Функция для вычисления числа обусловленности матрицы c нормой 2*/
template <typename T>
T cond_2(const vector<vector<T>>& matrix) {
    T n_1 = norm_2(matrix);
    if (n_1 == 0) {
        printf("Error: Det(A) = 0  =>  cond_2(A) = oo");
        return numeric_limits<T>::infinity();
    }
    vector<vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_2(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}

/* Функция для вычисления числа обусловленности матрицы с нормой oo*/
template <typename T>
T cond_oo(const vector<vector<T>>& matrix) {
    T n_1 = norm_oo(matrix);
    if (n_1 == 0) {
        printf("Error: Det(A) = 0  =>  cond_oo(A) = oo");
        return numeric_limits<T>::infinity();
    }
    vector<vector<T>> inverse_matrix = inverseMatrix(matrix);
    T n_2 = norm_oo(inverse_matrix);
    T cond = n_1 * n_2;
    return cond;
}


template <typename T>
vector<vector<T>> transpose(const vector<vector<T>>& A) {
    int rows = A.size();
    int cols = A[0].size();
    vector<vector<T>> result(cols, vector<T>(rows));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            result[j][i] = A[i][j];
        }
    }

    return result;
}

// Функция обрезки матрицы снизу и справа
template <typename T>
vector<vector<T>> crop_matrix(const vector<vector<T>>& A, const int& k) {

    int n = A.size();
    vector<vector<T>> A_crop(n - k, vector<T>(n - k, 0));
    for (int i = 0; i < (n - k); i++) {
        for (int j = 0; j < (n - k); j++) {
            A_crop[i][j] = A[i][j];
        }
    }

    return A_crop;
}

/* Функция, вычисляющая определитель матрицы 4х4 */
template <typename T>
double det(const vector<vector<T>>& matrix) {
    return
        matrix[0][0] * (
            matrix[1][1] * (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
            matrix[1][2] * (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) +
            matrix[1][3] * (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1])
            ) -
        matrix[0][1] * (
            matrix[1][0] * (matrix[2][2] * matrix[3][3] - matrix[2][3] * matrix[3][2]) -
            matrix[1][2] * (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
            matrix[1][3] * (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0])
            ) +
        matrix[0][2] * (
            matrix[1][0] * (matrix[2][1] * matrix[3][3] - matrix[2][3] * matrix[3][1]) -
            matrix[1][1] * (matrix[2][0] * matrix[3][3] - matrix[2][3] * matrix[3][0]) +
            matrix[1][3] * (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0])
            ) -
        matrix[0][3] * (
            matrix[1][0] * (matrix[2][1] * matrix[3][2] - matrix[2][2] * matrix[3][1]) -
            matrix[1][1] * (matrix[2][0] * matrix[3][2] - matrix[2][2] * matrix[3][0]) +
            matrix[1][2] * (matrix[2][0] * matrix[3][1] - matrix[2][1] * matrix[3][0]));
}

/* Функция, сортирующая вектор */
template< typename T>
vector<T> sorted(const vector<T>& vec_not_sort) {
    vector<T> vec(vec_not_sort);
    int n = vec.size();
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (vec[j] > vec[j + 1]) {
                // Обмен элементов, если они не упорядочены
                T temp = vec[j];
                vec[j] = vec[j + 1];
                vec[j + 1] = temp;
            }
        }
    }
    return vec;
}

/* Функция, возращающая максимальный по модулю элемент вектора */
template<typename T>
T vec_max(const vector<T>& vec) {
    int n = vec.size();
    T max = 0;
    for (int i = 0; i < n; i++) {
        if (abs(vec[i]) > max)
            max = abs(vec[i]);
    }
    return max;
}

template<typename T>
T test_eigen(const vector<vector<T>>& matrix, const vector<T>& lambda) {
    int n = matrix.size();
    vector<vector<T>> A(matrix);
    vector<vector<T>> E = create_identity_matrix<T>(n);

    T max_error = 0;
    T min_error = 0;
    for (int i = 0; i < n; i++) {
        vector<vector<T>> A_test = A - lambda[i] * E;
        T error = det(A_test);
        max_error = (max_error >= error) ? max_error : error;
        min_error = (min_error <= error) ? min_error : error;
    }
    return max_error;
}

/* Функция, возращающая норму разницы решений */
template<typename T>
T testeps(const vector<T>& x, const vector<T>& true_x, const int p) {
    vector<T> delta = x - true_x;
    return norm(delta, p);
}

template<typename T>
T test_eigen_vec(const vector<vector<T>>& matrix, vector<vector<T>> eigen_vec, const vector<T>& lambda) {
    int n = matrix.size();
    vector<vector<T>> A(matrix);
    vector<vector<T>> E = create_identity_matrix<T>(n);

    T max_error = 0;
    T min_error = 10e10;
    for (int i = 0; i < n; i++) {
        vector<T> A_test = A * eigen_vec[i];
        A_test = A_test - lambda[i] * eigen_vec[i];
        T error = norm(A_test, 2);
        max_error = (max_error >= error) ? max_error : error;
        min_error = (min_error <= error) ? min_error : error;
    }

    return max_error;
}

template<typename T>
T test_eigen_vec2(const vector<vector<T>>& matrix, vector<T> eigen_vec, const T lambda) {
    int n = matrix.size();
    vector<vector<T>> A(matrix);
    vector<vector<T>> E = create_identity_matrix<T>(n);

    vector<T> A_test = A * eigen_vec;
    A_test = A_test - lambda * eigen_vec;
    T error = norm(A_test, 2);


    return error;
}


template<typename T>
T cos_vec(const vector<T> x1, const vector<T> x2) {

    T scalar = dot(x1, x2);
    T norm1 = norm(x1, 2);
    T norm2 = norm(x2, 2);

    return scalar / (norm1 * norm2);

}

template<typename T>
vector<vector<T>> uncrop(vector<vector<T>> A, vector<vector<T>>A1, int crop) {
    int n = A1.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = A1[i][j];
        }
    }
    return A;
}



// Функция для LU-разложения с частичным выбором
template <typename T>
void lu_decomposition(const vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& U) {
    int n = A.size();
    for (int i = 0; i < n; i++) {
        int pivot_row = i;
        T max_val = 0;
        for (int k = i; k < n; k++) {
            if (abs(A[k][i]) > max_val) {
                max_val = abs(A[k][i]);
                pivot_row = k;
            }
        }

        if (pivot_row != i) {
            swap(A[i], A[pivot_row]);
            swap(L[i], L[pivot_row]);
            swap(U[i], U[pivot_row]);
        }

        for (int j = i; j < n; j++) {
            U[i][j] = A[i][j];
            L[i][j] = i == j ? 1 : 0;
            for (int k = 0; k < i; k++) {
                U[i][j] -= L[i][k] * U[k][j];
            }
        }

        for (int j = i + 1; j < n; j++) {
            L[j][i] = A[j][i];
            for (int k = 0; k < i; k++) {
                L[j][i] -= L[j][k] * U[k][i];
            }
            L[j][i] /= U[i][i];
        }
    }
}

/* Функция для вычисления нормы вектора невязки */
template <typename T>
T norm_vector_nevazki(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x, const int& n) {
    int s = A.size();
    vector<T> residual(s, 0);


    for (int i = 0; i < s; i++) {
        for (int j = 0; j < s; j++) {
            residual[i] += A[i][j] * x[j];
        }
        residual[i] = b[i] - residual[i];
    }
    T residual_norm = norm(residual, n);
    return residual_norm;
}

/* Функция для решения СЛАУ прямым методом Гаусса */
template <typename T>
vector<T> method_Gaussa(const vector<vector<T>>& matrix, const vector<T>& vec, const T& eps) {
    int n = matrix.size();

    // Создаем копии матрицы и вектора
    vector<vector<T>> A(matrix);
    vector<T> b(vec);

    // Прямой ход
    for (int i = 0; i < n; i++) {
        // Поиск максимального элемента в текущем столбце и его индекса
        int maxRow = i;
        T maxVal = fabs(A[i][i]);
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > maxVal) {
                maxVal = fabs(A[k][i]);
                maxRow = k;
            }
        }

        if (maxVal < eps) {
            printf("Error: Det(matrix) = 0 \n");
            exit(1);
        }

        // Обмен строк, если необходимо
        if (maxRow != i) {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
        }

        // Делаем текущий диагональный элемент равным 1
        T a = A[i][i];
        for (int j = i; j < n; j++) {
            A[i][j] /= a;
        }
        b[i] /= a;

        // Обнуляем элементы под текущим диагональным элементом
        for (int k = i + 1; k < n; k++) {
            T factor = A[k][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Подстановка обратно в систему
    vector<T> x(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
    }

    return x;

}

/* Функция для решения СЛАУ прямым методом Гаусса для 4 лабы, без вывода ошибки о нулевом определителе */
template <typename T>
vector<T> method_Gaussa2(const vector<vector<T>>& matrix, const vector<T>& vec, const T& eps) {
    int n = matrix.size();

    // Создаем копии матрицы и вектора
    vector<vector<T>> A(matrix);
    vector<T> b(vec);

    // Прямой ход
    for (int i = 0; i < n; i++) {
        // Поиск максимального элемента в текущем столбце и его индекса
        int maxRow = i;
        T maxVal = fabs(A[i][i]);
        for (int k = i + 1; k < n; k++) {
            if (fabs(A[k][i]) > maxVal) {
                maxVal = fabs(A[k][i]);
                maxRow = k;
            }
        }

        //        if (maxVal < eps) {
        //            printf("Error: Det(matrix) = 0 \n");
        //            exit(1);
        //        }

                // Обмен строк, если необходимо
        if (maxRow != i) {
            swap(A[i], A[maxRow]);
            swap(b[i], b[maxRow]);
        }

        // Делаем текущий диагональный элемент равным 1
        T a = A[i][i];
        for (int j = i; j < n; j++) {
            A[i][j] /= a;
        }
        b[i] /= a;

        // Обнуляем элементы под текущим диагональным элементом
        for (int k = i + 1; k < n; k++) {
            T factor = A[k][i];
            for (int j = i; j < n; j++) {
                A[k][j] -= factor * A[i][j];
            }
            b[k] -= factor * b[i];
        }
    }

    // Подстановка обратно в систему
    vector<T> x(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = b[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
    }

    return x;

}

/* Функция для оценки изменения числа обуcловленности от возмущения вектора правой части через метод Гаусса */
template <typename T>
void min_change_cond(const vector<vector<T>>& matrix, const  vector<T>& vec, const vector<T>& mod) {
    /* Находим минимальное значение числа обусловленности */

    // Находим относительную погрешность
    T delta_b_1 = norm(mod, 1) / norm(vec, 1);
    T delta_b_2 = norm(mod, 2) / norm(vec, 2);
    T delta_b_oo = norm(mod, 0) / norm(vec, 0);

    // Находим относительную погрешность x
    T delta_x_1 = 0;
    T delta_x_2 = 0;
    T delta_x_oo = 0;

    vector<T> solve = method_Gaussa(matrix, vec);
    vector<T> mod_vec;

    for (int epo = 0; epo < 50; epo++) {

        // создаем модифицированный вектор правой части
        mod_vec = vec;

        for (int i = 0; i < mod_vec.size(); i++) {
            mod_vec[i] += mod[i] * pow(-1, rand());
        }

        // Ищем максимальное изменение нормы вектора изменения решения
        vector<T> mod_solve = method_Gaussa(matrix, mod_vec);

        for (int i = 0; i < mod_solve.size(); i++) {
            mod_solve[i] = abs(mod_solve[i] - solve[i]);
        }
        delta_x_1 = (delta_x_1 <= norm(mod_solve, 1)) ? norm(mod_solve, 1) : delta_x_1;
        delta_x_2 = (delta_x_2 <= norm(mod_solve, 2)) ? norm(mod_solve, 2) : delta_x_2;
        delta_x_oo = (delta_x_oo <= norm(mod_solve, 0)) ? norm(mod_solve, 0) : delta_x_oo;
    }

    delta_x_1 /= norm(solve, 1);
    delta_x_2 /= norm(solve, 2);
    delta_x_oo /= norm(solve, 0);

    T min_cond_1 = delta_x_1 / delta_b_1;
    T min_cond_2 = delta_x_2 / delta_b_2;
    T min_cond_oo = delta_x_oo / delta_b_oo;

    /* Находим максимальное значение числа обусловленности */

    int n = matrix.size();

    vector<vector<T>> U(matrix);
    vector<vector<T>> L(matrix);

    lu_decomposition(matrix, L, U);

    L = transpose(L);
    T max_cond_1 = cond_1(L) * cond_1(U);
    T max_cond_2 = cond_2(L) * cond_2(U);
    T max_cond_oo = cond_oo(L) * cond_oo(U);


    cout << endl;
    cout << min_cond_1 << " <= cond_1(A) <= " << max_cond_1 << endl;
    cout << min_cond_2 << " <= cond_2(A) <= " << max_cond_2 << endl;
    cout << min_cond_oo << " <= cond_oo(A) <= " << max_cond_oo << endl;
    cout << endl;
}


/* Функция QR-разложения матрицы методом вращений */
template<typename T>
void QR_decomposition(const vector<vector<T>>& matrix, vector<vector<T>>& Q, vector<vector<T>>& R, const T& eps) {

    int n = matrix.size();
    R = matrix;                        // R - копия матрицы A
    Q = create_identity_matrix<T>(n);  // Q - единичная матрица
    T c, s;                            // Коэффициенты с и s

    for (int i = 0; i < n; ++i) {
        int m = i;
        for (int k = i; k < n; ++k) {
            if (fabs(R[k][i]) > fabs(R[m][i])) {
                m = k;
            };
        }

        for (int k = 0; k < n; ++k) {
            swap(R[m][k], R[i][k]);
            swap(Q[k][m], Q[k][i]);
        }

        //        if (abs(R[i][i]) <= eps){
        //            cout << "Error in QR_decomposition" << endl;
        //            system("pause");
        //            exit(1);
        //        }
        if (abs(R[i][i]) < eps) {
            c = 1; // Тождественное преобразование, поворот на 0 градусов
            s = 0; // Нет необходимости избавляться от этих переменных
        }

        for (int j = i + 1; j < n; ++j) {
            c = (R[i][i]) / sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]);
            s = (R[j][i]) / sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]);

            for (int k = 0; k < n; ++k) {
                T aa = R[i][k];
                T ab = R[j][k];

                R[i][k] = c * aa + s * ab;
                R[j][k] = c * ab - s * aa;

                T qa = Q[k][i];
                T qb = Q[k][j];

                Q[k][i] = c * qa + s * qb;
                Q[k][j] = c * qb - s * qa;
            }
            R[j][i] = 0;
        }
    }
}

/* Функция QR-разложения уменьшенной матрицы*/
template <typename T>
void QR_decomposition_with_crop(const vector<vector<T>>& A, vector<vector<T>>& Q, vector<vector<T>>& R, const T& eps) {
    R = A;
    int n = A.size();
    Q = create_identity_matrix<T>(A.size());
    T c, s; // Kоэффициенты матрицы поворота

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (abs(R[i][i]) < eps || abs(R[i][j]) < eps) {
                c = 1; // Поворот на 0 градусов
                s = 0;
            }
            else {
                c = R[i][i] / (sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]));
                s = R[j][i] / (sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]));

                for (int k = 0; k < n; ++k) {
                    T buf;
                    buf = c * R[i][k] + s * R[j][k];
                    R[j][k] = c * R[j][k] - s * R[i][k];
                    R[i][k] = buf;

                    buf = c * Q[i][k] + s * Q[j][k];
                    Q[j][k] = c * Q[j][k] - s * Q[i][k];
                    Q[i][k] = buf;
                }
            }
        }
    }
    Q = transpose(Q);
}

/* Функция QR-разложения уменьшенной матрицы*/
template <typename T>
void QR_decomposition_with_crop(const vector<vector<T>>& A, vector<vector<T>>& Q, vector<vector<T>>& R, const int crop, const T& eps) {
    R = A;
    Q = create_identity_matrix<T>(A.size());
    T c, s; // Kоэффициенты матрицы поворота

    for (int i = 0; i < crop; ++i) {
        for (int j = i + 1; j < crop; ++j) {
            if (abs(R[i][i]) < eps || abs(R[i][j]) < eps) {
                c = 1; // Поворот на 0 градусов
                s = 0;
            }
            else {
                c = R[i][i] / (sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]));
                s = R[j][i] / (sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]));

                for (int k = 0; k < crop; ++k) {
                    T buf;
                    buf = c * R[i][k] + s * R[j][k];
                    R[j][k] = c * R[j][k] - s * R[i][k];
                    R[i][k] = buf;

                    buf = c * Q[i][k] + s * Q[j][k];
                    Q[j][k] = c * Q[j][k] - s * Q[i][k];
                    Q[i][k] = buf;
                }
            }
        }
    }
    Q = transpose(Q);
}

/* Функция решения СЛАУ методом QR */
template <typename T>
vector<T> method_QR(const vector<vector<T>>& A, const vector<T>& b, const T& eps) {
    int n = A.size();
    vector<vector<T>> Q, R;
    QR_decomposition(A, Q, R, eps);

    // Решение системы Q^T * y = b
    vector<T> y(n, 0);
    for (int i = 0; i < n; i++) {
        T dotProduct = 0;
        for (int j = 0; j < n; j++) {
            dotProduct += Q[j][i] * b[j];
        }
        y[i] = dotProduct;
    }

    // Решение системы R * x = y методом обратной подстановки
    vector<T> x(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

/* Метод QR с известным разложением */
template <typename T>
vector<T> method_QR_withoutQR(const vector<vector<T>>& A, const vector<T>& b, const vector<vector<T>>& Q, const vector<vector<T>>& R, const T& eps) {

    int n = A.size();
    // Решение системы Q^T * y = b
    vector<T> y(n, 0);
    for (int i = 0; i < n; i++) {
        T dotProduct = 0;
        for (int j = 0; j < n; j++) {
            dotProduct += Q[j][i] * b[j];
        }
        y[i] = dotProduct;
    }

    // Решение системы R * x = y методом обратной подстановки
    vector<T> x(n, 0);
    for (int i = n - 1; i >= 0; i--) {
        x[i] = y[i];
        for (int j = i + 1; j < n; j++) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }

    return x;
}

/* Функция для оценки изменения числа обуcловленности от возмущения вектора правой части через метод QR */
template <typename T>
void min_change_cond2(const vector<vector<T>>& matrix, const vector<T>& vec, const vector<T>& mod, const T& eps) {
    /* Находим минимальное значение числа обусловленности */

    // Находим относительную погрешность vec
    T delta_b_1 = norm_1(mod) / norm_1(vec);
    T delta_b_2 = norm_1(mod) / norm_1(vec);
    T delta_b_oo = norm_1(mod) / norm_1(vec);

    // Находим относительную погрешность x
    T delta_x_1 = 0;
    T delta_x_2 = 0;
    T delta_x_oo = 0;

    vector<T> solve = method_Gaussa(matrix, vec, eps);

    vector<T> mod_vec = vec;
    vector<vector<T>> all_mod_vec = generateCombinations(mod);

    int n = matrix.size();
    vector<vector<T>> Q, R;
    QR_decomposition(matrix, Q, R, eps);

    for (int k = 0; k < all_mod_vec.size(); k++) {
        // Cоздаем модифицированный вектор правой части
        mod_vec = vec_sum(mod_vec, all_mod_vec[k]);
        // Ищем максимальное изменение нормы вектора изменения решения
        vector<T> mod_solve = method_QR_withoutQR(matrix, mod_vec, Q, R, eps);

        for (int i = 0; i < mod_solve.size(); i++) {
            mod_solve[i] = abs(mod_solve[i] - solve[i]);
        }
        delta_x_1 = (delta_x_1 <= norm_1(mod_solve)) ? norm_1(mod_solve) : delta_x_1;
        delta_x_2 = (delta_x_2 <= norm_2(mod_solve)) ? norm_2(mod_solve) : delta_x_2;
        delta_x_oo = (delta_x_oo <= norm_oo(mod_solve)) ? norm_oo(mod_solve) : delta_x_oo;
    }


    delta_x_1 /= norm_1(solve);
    delta_x_2 /= norm_1(solve);
    delta_x_oo /= norm_1(solve);

    T min_cond_1 = delta_x_1 / delta_b_1;
    T min_cond_2 = delta_x_2 / delta_b_2;
    T min_cond_oo = delta_x_oo / delta_b_oo;

    /* Находим максимальное значение числа обусловленности */

    vector<vector<T>> U(matrix);
    vector<vector<T>> L(matrix);

    lu_decomposition(matrix, L, U);

    L = transpon(L);
    T max_cond_1 = cond_1(L) * cond_1(U);
    T max_cond_2 = cond_2(L) * cond_2(U);
    T max_cond_oo = cond_oo(L) * cond_oo(U);


    cout << endl;
    cout << min_cond_1 << " <= cond_1(A) <= " << max_cond_1 << endl;
    cout << min_cond_2 << " <= cond_2(A) <= " << max_cond_2 << endl;
    cout << min_cond_oo << " <= cond_oo(A) <= " << max_cond_oo << endl;
    cout << endl;
}

/* ### Функций лабы 2 ### */


/* Функция представления матрицы С в виде: C = C_l + C_d + D_u (Нижнетреугольной, Диагональной, Верхнетреугольной) */
template<typename T>
void LDU_decomposotion(const vector<vector<T>>& A, vector<vector<T>>& L, vector<vector<T>>& D, vector<vector<T>>& U) {
    int n = A.size();

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i < j) {
                U[i][j] = A[i][j];
            }
            else if (i > j) {
                L[i][j] = A[i][j];
            }
            else {
                D[i][j] = A[i][j];
            }
        }
    }
}

/* Функция исследования итерационного параметра tau для метода простых итераций (Метод Золотого сечения)*/
template<typename T>
T SimpleIterations_method_matrix_norm_C(const vector<vector<T>>& A, const T& tau, const int& p) {
    vector<vector<T>> E = create_identity_matrix<T>(A.size()); // Единичный вектор
    vector<vector<T>> C = -(tau * A - E);                   // Матрица С
    return norm(C, p);
}

// Метод золотого сечения для поиска минимума функции на заданном интервале [a, b]
template<typename T>
T golden_section_search_tau(const vector<vector<T>>& A, T a, T b, const int& p, const T& epsilon) {

    const T golden_ratio = 1.618033988749895; // Золотое Сечение
    T x1, x2;

    // Начальные точки
    x1 = b - (b - a) / golden_ratio;
    x2 = a + (b - a) / golden_ratio;

    while (fabs(b - a) > epsilon) {
        if (SimpleIterations_method_matrix_norm_C(A, x1, p) < SimpleIterations_method_matrix_norm_C(A, x2, p)) {
            b = x2;
            x2 = x1;
            x1 = b - (b - a) / golden_ratio;
        }
        else {
            a = x1;
            x1 = x2;
            x2 = a + (b - a) / golden_ratio;
        }
    }
    return (a + b) / 2;
}


template<typename T>
MyResult2<T> method_SimpleIteration(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& tau, const T& eps, const int& p, const int& MaxIter) {

    MyResult2<T> result;
    vector<vector<T>> E = create_identity_matrix<T>(A.size());   // Единичный вектор
    vector<vector<T>> C = -(tau * A - E);                           // Матрица С
    vector<T> y = tau * b;                                          // Вектор y
    vector<T> xk = x0;
    vector<T> xk_new = xk;

    result.C = C;
    result.y = y;

    for (int i = 0; i < MaxIter; ++i) {

        xk_new = C * xk + y;

        // Критерий останова итерационного процесса
        vector<T> delta_stop = xk_new - xk;
        xk = xk_new;
        if ((norm_vector_nevazki(A, b, xk, p) <= eps) or (norm(delta_stop, p) <= (((1 - norm(C, p)) / norm(C, p)) * eps))) {
            result.solve = xk;
            result.iterations = i + 1;
            result.batch = norm(delta_stop, p);
            return result;
        }
    }
    result.solve = xk;
    result.iterations = MaxIter;
    return result;
}


/* Функция решения СЛАУ методом Якоби */
template<typename T>
MyResult2<T> method_Yacobi(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& eps, const int& p, const int& MaxIter) {

    MyResult2<T> result;
    vector<vector<T>> L(A.size(), vector<T>(A.size(), 0)), D(A.size(), vector<T>(A.size(), 0)), U(A.size(), vector<T>(A.size(), 0));
    LDU_decomposotion(A, L, D, U);

    vector<vector<T>> D_inv = inverseMatrix(D);
    vector<vector<T>> C = -D_inv * (L + U);
    vector<T> y = D_inv * b;

    result.C = C;
    result.y = y;

    vector<T> xk = x0;
    vector<T> xk_new = xk;
    T C_norm = norm(C, p);



    for (int i = 0; i < MaxIter; ++i) {

        xk_new = C * xk + y;

        // Критерий останова итерационного процесса
        vector<T> delta_stop = xk_new - xk;
        xk = xk_new;
        if ((norm_vector_nevazki(A, b, xk, 1) <= eps) or (norm(delta_stop, p) <= (((1 - norm(C, p)) / norm(C, p)) * eps))) {
            result.solve = xk;
            result.iterations = i + 1;
            result.batch = norm(delta_stop, p);
            return result;
        }
    }
    result.solve = xk;
    result.iterations = MaxIter;
    return result;

}
template <typename T>
T C_matrix_for_relax(const vector<vector<T>>& A, const T& w, const int& p) {

    vector<vector<T>> L(A.size(), vector<T>(A.size(), 0)), D(A.size(), vector<T>(A.size(), 0)), U(A.size(), vector<T>(A.size(), 0));
    LDU_decomposotion(A, L, D, U);
    vector<vector<T>> C = ((1 / w) * D - L);
    C = inverseMatrix(C);
    C = C * (((1 - w) / w) * D + U);
    return norm(C, p);
}

/* Функция исследования итерационного параметра W для метода Релаксации для трехдиагональной матрицы (Метод Золотого сечения)*/
template<typename T>
T golden_section_search_W(const vector<vector<T>>& A, T a, T b, const int& p, const T& eps) {

    const T golden_ratio = 1.618033988749895; // Золотое Сечение
    T x1, x2;

    // Начальные точки
    x1 = b - (b - a) / golden_ratio;
    x2 = a + (b - a) / golden_ratio;

    while (abs(b - a) > eps) {
        if (C_matrix_for_relax(A, x1, p) < C_matrix_for_relax(A, x2, p)) {
            b = x2;
            x2 = x1;
            x1 = b - (b - a) / golden_ratio;
        }
        else {
            a = x1;
            x1 = x2;
            x2 = a + (b - a) / golden_ratio;
        }
    }
    return (a + b) / 2;
}

/* Функция решения СЛАУ методом Релаксации через каноническую формулу */
template <typename T>
MyResult2<T> method_Relax(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& w, const T& eps, const int& p, const int& MaxIter) {
    MyResult2<T> result;

    int n = A.size();
    vector<T> x = x0;  // Начальное приближение

    // Вычисление ненужной матрицы С для вывода
    vector<vector<T>> L(A.size(), vector<T>(A.size(), 0)), D(A.size(), vector<T>(A.size(), 0)), U(A.size(), vector<T>(A.size(), 0));
    LDU_decomposotion(A, L, D, U);
    vector<vector<T>> C = ((1 / w) * D - L);
    C = inverseMatrix(C);
    vector<T> y = C * b;
    C = C * (((1 - w) / w) * D + U);

    result.C = C;
    result.y = y;

    for (int k = 0; k < MaxIter; ++k) {
        vector<T> x_new(n, 0);

        for (int i = 0; i < n; ++i) {
            T sum1 = 0;
            T sum2 = 0;

            for (int j = 0; j < i; ++j) {
                sum1 += A[i][j] * x_new[j];
            }

            for (int j = i + 1; j < n; ++j) {
                sum2 += A[i][j] * x[j];
            }

            x_new[i] = (1 - w) * x[i] + (w / A[i][i]) * (b[i] - sum1 - sum2);
        }

        // Критерий останова итерационного процесса
        vector<T> delta_stop = x_new - x;
        x = x_new;
        if ((norm_vector_nevazki(A, b, x, p) <= eps) or
            (norm(delta_stop, p) <= (((1 - norm(C, p)) / norm(C, p)) * eps))) {
            result.solve = x;
            result.iterations = k + 1;
            result.batch = norm(delta_stop, p);
            return result;
        }
    }
    // Если не достигнута необходимая точность за максимальное число итераций
    result.solve = x;
    result.iterations = MaxIter;
    return result;

}

/* Функция решения СЛАУ методом Релаксации через матричную формулу */
//template<typename T>
//Result<T> method_Relax(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& w, const T& eps, const int& p, const int& MaxIter){
//    Result<T> result;
//
//    vector<vector<T>> L(A.size(), vector<T>(A.size(), 0)),D(A.size(), vector<T>(A.size(), 0)), U(A.size(), vector<T>(A.size(), 0));
//    LDU_decomposotion(A, L, D, U);
//
//    vector<vector<T>> C = ((1 / w) * D - L);
//    C = inverseMatrix(C);
//    vector<T> y = C * b;
//    C = C * (((1 - w) / w) * D + U);
//
//    vector<T> xk = x0;
//    vector<T> xk_new = xk;
//
//    result.C = C;
//
//    for (int i = 0; i < MaxIter; ++i){
//
//        xk_new = C * xk + y;
//
//        // Критерий останова итерационного процесса
//        vector<T> delta_stop = xk_new - xk;
//        xk = xk_new;
//        if ((norm_vector_nevazki(A, b, xk, p) <= eps) or (norm(delta_stop, p) <= (((1 - norm(C, p)) / norm(C, p)) * eps))) {
//            result.solve = xk;
//            result.iterations = i + 1;
//            return result;
//        }
//    }
//    result.solve = xk;
//    result.iterations = MaxIter;
//    return result;
//}


/* Функция решения СЛАУ методом Зейделя */
template <typename T>
MyResult2<T> method_Zeidel(const vector<vector<T>>& A, const vector<T>& b, const vector<T>& x0, const T& eps, const int& p, const int& MaxIter) {
    MyResult2<T> result = method_Relax<T>(A, b, x0, 1, eps, p, MaxIter);
    return result;
}


/* Функция для вычисления нормы вектора невязки трехдиагональной СЛАУ */
template <typename T>
T norm_vector_nevazki(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& solution, const int& n) {
    // Вычисление невязки
    vector<T> residual(n);
    for (int i = 0; i < n; ++i) {
        residual[i] = b[i] - (A[i - 1] * solution[i - 1] + B[i] * solution[i] + C[i] * solution[i + 1]);
    }

    // Норма невязки
    T residual_norm = norm(residual, n);
    return residual_norm;
}


/* Функция решения трехдиагональной СЛАУ большой размерности методом Зейделя */
template <typename T>
vector<T> method_Zeidel_diag(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& x0, const T& eps, const T& maxIterations) {
    size_t n = A.size();
    vector<T> x = x0; // Начальное приближение

    for (int iter = 0; iter < maxIterations; ++iter) {
        T max_diff = 0;

        for (size_t i = 0; i < n; ++i) {
            T sum1 = 0.0;
            if (i > 0) {
                sum1 = A[i] * x[i - 1];
            }

            T sum2 = 0.0;
            if (i < n - 1) {
                sum2 = C[i] * x[i + 1];
            }

            T new_x_i = (1 - 1) * x[i] + (1 / B[i]) * (b[i] - sum1 - sum2);

            max_diff = max(max_diff, abs(new_x_i - x[i]));

            x[i] = new_x_i;
        }

        // Проверка на достижение необходимой точности
        if (max_diff < eps) {
            cout << "Method_Zeidel_diag converged after " << iter + 1 << " iterations\n";
            return x;
        }
    }
    cout << "Method_Zeidel_diag DON'T converged after " << maxIterations << " iterations\n";
    return x;
}



/* Функция решения трехдиагональной СЛАУ большой размерности методом Релаксации */
template <typename T>
vector<T> method_Relax_diag(const vector<T>& A, const vector<T>& B, const vector<T>& C, const vector<T>& b, const vector<T>& x0, const T& w, const T& eps, const T& MaxIter) {
    size_t n = A.size();
    vector<T> x = x0; // Начальное приближение

    // Итерации метода релаксации
    for (int iter = 0; iter < MaxIter; ++iter) {
        T max_diff = 0;

        for (size_t i = 0; i < n; ++i) {
            T sum = 0;

            if (i > 0) {
                sum += A[i] * x[i - 1];
            }

            if (i < n - 1) {
                sum += C[i] * x[i + 1];
            }

            T new_x_i = (1 - w) * x[i] + (w / B[i]) * (b[i] - sum);

            max_diff = max(max_diff, abs(new_x_i - x[i]));

            x[i] = new_x_i;
        }

        // Проверка на достижение необходимой точности
        if (max_diff < eps) {
            cout << "Method_Relax_diag converged after " << iter + 1 << " iterations\n";
            return x;
        }
    }
    cout << "Method_Relax_diag DON'T converged after " << MaxIter << " iterations\n";
    return x;
}


//* Функция исследования итерационного параметра W для метода Релаксации для трехдиагональной матрицы (Метод Золотого сечения)*/
template<typename T>
T golden_section_search_W(vector<T> A, vector<T> B, vector<T> C, vector<T> vec, vector<T> x, T EPS, int MaxIteration, T a, T b) {

    const T golden_ratio = 1.618033988749895; // Золотое Сечение
    T x1, x2;

    // Начальные точки
    x1 = b - (b - a) / golden_ratio;
    x2 = a + (b - a) / golden_ratio;

    while (abs(b - a) > EPS) {

        vector<T> sol1 = method_Relax_diag(A, B, C, vec, x, x1, EPS, MaxIteration);
        vector<T> sol2 = method_Relax_diag(A, B, C, vec, x, x2, EPS, MaxIteration);
        T norm_sol1 = norm_vector_nevazki(A, B, C, vec, sol1, 1);
        T norm_sol2 = norm_vector_nevazki(A, B, C, vec, sol2, 1);



        if (norm_sol1 < norm_sol2) {
            b = x2;
            x2 = x1;
            x1 = b - (b - a) / golden_ratio;
        }
        else {
            a = x1;
            x1 = x2;
            x2 = a + (b - a) / golden_ratio;
        }
    }
    return (a + b) / 2;
}


/* Функция, которая делает диагональные элементы матрицы больше нуля */

template<typename T>
vector<vector<T>> make_plus_diaf_matrix(vector<vector<T>> matrix) {
    int rows = matrix.size();
    int cols = matrix[0].size();


    vector<vector<T>> result(rows, vector<T>(cols, 0));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Увеличиваем диагональные элементы
            if (i == j) {
                result[i][j] = matrix[i][j] + 1;
            }
            else {
                result[i][j] = matrix[i][j];
            }
        }
    }

    return result;
}



/* Функция априорной оценки */
template <typename T>
void aprior_eps(const vector<vector<T>>& C, const vector<T>& y, const vector<T>& x0, const int& p) {
    T normC = norm(C, p);
    vector<T> delta_stop = C * x0 + y;
    T norm_delta_stop = norm(delta_stop, p);
    cout << "Aprior Eps <= " << abs((normC / (1 - normC)) * norm_delta_stop) << endl;
}


/* Функция апостериорной оценки */
template <typename T>
void aposter_eps(const vector<vector<T>>& C, T norm_delta, const int& p) {
    T normC = norm(C, p);
    cout << "Aposterior Eps <= " << abs((normC / (1 - normC)) * norm_delta) << endl;
}




/* ### Функций лабы 4 ### */

/* Функция приведения матрицы к форме Хессенберга методом вращений */
template <typename T>
vector<vector<T>> Hessenberg_decomposition(const vector<vector<T>>& matrix, const T& eps) {
    vector<vector<T>> A = matrix; // Результирующая матрица

    int m = A.size();

    T c, s; // Коэффициенты матрицы поворота
    T buf;  // Обменник

    for (int i = 0; i < m - 2; ++i)
    {
        for (int j = i + 2; j < m; ++j)
        {
            if (abs(A[i + 1][i]) < eps || abs(A[j][i]) < eps)  // Если R[i + 1][i] = 0 или R[j][i + 1] = 0
            {
                c = 1; // Поворот на 0 градусов
                s = 0;
            }
            else
            {
                c = A[i + 1][i] / (sqrt(A[i + 1][i] * A[i + 1][i] + A[j][i] * A[j][i]));
                s = A[j][i] / (sqrt(A[i + 1][i] * A[i + 1][i] + A[j][i] * A[j][i]));
                for (int k = i; k < m; ++k)
                {
                    buf = A[i + 1][k];
                    A[i + 1][k] = c * A[i + 1][k] + s * A[j][k];
                    A[j][k] = c * A[j][k] - s * buf;
                }

                for (int k = i; k < m; ++k)
                {
                    buf = A[k][i + 1];
                    A[k][i + 1] = c * A[k][i + 1] + s * A[k][j];
                    A[k][j] = c * A[k][j] - s * buf;
                }
            }
        }
    }

    return A;
}

template<typename T>
void transposemultiply(vector<vector<T>>& A, int k, int l, int M, T a, T b, int& buf) {
    T tmp;
    for (int i = k; i < M; i++) {
        tmp = A[k + 1][i];
        A[k + 1][i] = a * A[k + 1][i] + b * A[l][i];
        A[l][i] = -b * tmp + a * A[l][i];
        buf += 4;
    }
    for (int i = k; i < M; i++) {
        tmp = A[i][k + 1];
        A[i][k + 1] = a * A[i][k + 1] + b * A[i][l];
        A[i][l] = -b * tmp + a * A[i][l];
        buf += 4;
    }
}

/* Функция приведения матрицы к форме Хессенберга методом вращений */
template <typename T>
vector<vector<T>> Hessenberg_decomposition2(const vector<vector<T>>& matrix, const T& eps) {
    vector<vector<T>> A = matrix; // Результирующая матрица
    int n = A.size();
    double d;
    double alpha;
    double betha;
    int mults = 0;

    for (int i = 0; i < n - 2; i++) {
        for (int j = i + 2; j < n; j++) {
            mults += 4;
            d = sqrt(A[i + 1][i] * A[i + 1][i] + A[j][i] * A[j][i]);
            alpha = A[i + 1][i] / d;
            betha = A[j][i] / d;
            transposemultiply(A, i, j, n, alpha, betha, mults);
        }
    }
    return A;
}


/* Функция нахождения собственных значений матрицы методом QR-разложения */
template <typename T>
MyResult4<T> Eigen_method_QR(const vector<vector<T>>& matrix, const T& sigma, const T& eps, const int& maxIterations) {
    MyResult4<T> result;
    result.iterations = maxIterations;

    int n = matrix.size();
    vector<vector<T>> A(matrix);
    vector<vector<T>> E = create_identity_matrix<T>(n);
    vector<vector<T>> Q, R;
    T max_elem = eps;
    int iter = 0;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < maxIterations; j++) {
            max_elem = 0; // Обнуление максимального по модулю элемента текущей строки слева A[i][i]
            iter += 1;
            // Алгоритм:
            A = A - sigma * E;                                              // Сдвиг
            T cropp = i + 1;
            QR_decomposition_with_crop(A, Q, R, cropp, eps);    // QR-разложение
            A = R * Q;                                                     // Обновление A
            A = A + sigma * E;                                             // Обратный сдвиг

            result.A_iter.push_back(A);
            // Ищем максимальный по модулю элемент в строке слева A[i][i]
            for (int j = 0; j < i; ++j) {
                if (abs(A[i][j]) > max_elem)
                    max_elem = abs(A[i][j]);
            }
            // Критерий останова максимальный по модулю элемент текущей строки слева A[i][i] меньше точности
            if (max_elem < eps) {
                break;
            }
        }
    }

    // Вектор собственных чисел
    vector<T> eigen(n, 0);
    for (int i = 0; i < n; i++) {
        eigen[i] = A[i][i];
    }

    result.eigen = sorted(eigen);
    result.R = A;
    result.iterations = iter;
    return result;
}

/* Функция нахождения собственных значений матрицы методом QR-разложения */
template <typename T>
MyResult4<T> Eigen_method_QR(const vector<vector<T>>& matrix, const T& eps, const int& maxIterations) {
    MyResult4<T> result;
    result.iterations = maxIterations;

    int n = matrix.size();
    vector<vector<T>> A(matrix);
    vector<vector<T>> E = create_identity_matrix<T>(n);
    vector<vector<T>> Q, R;
    T max_elem = eps;
    int iter = 0;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < maxIterations; j++) {
            max_elem = 0; // Обнуление максимального по модулю элемента текущей строки слева A[i][i]
            iter += 1;
            T sigma = A[i][i];
            // Алгоритм:
            A = A - sigma * E;                                              // Сдвиг
            QR_decomposition_with_crop(A, Q, R, i + 1, eps);    // QR-разложение
            A = R * Q;                                                     // Обновление A
            A = A + sigma * E;                                             // Обратный сдвиг


            result.A_iter.push_back(A);

            // Ищем максимальный по модулю элемент в строке слева A[i][i]
            for (int j = 0; j < i; ++j) {
                if (abs(A[i][j]) > max_elem)
                    max_elem = abs(A[i][j]);
            }
            // Критерий останова максимальный по модулю элемент текущей строки слева A[i][i] меньше точности
            if (max_elem < eps) {
                break;
            }
        }
    }

    // Вектор собственных чисел
    vector<T> eigen(n, 0);
    for (int i = 0; i < n; i++) {
        eigen[i] = A[i][i];
    }

    result.eigen = sorted(eigen);
    result.R = A;
    result.iterations = iter;
    return result;
}



template <typename T>
MyResult4<T> Eigen_method_QR2(const vector<vector<T>>& matrix, const T& eps, const int& maxIterations) {

    MyResult4<T> result;
    vector<vector<T>> A(matrix);
    T n = A.size();
    vector<vector<T>> E = create_identity_matrix<T>(n);
    vector<vector<T>> V;
    int iter = 0;

    for (int i = 0; i < n; i++) {
        A = crop_matrix(A, i);
        print(A);
    }

    return result;
}


/* Функция нахождения собственных значений матрицы методом QR-разложения */
template <typename T>
MyResult4<T> Eigen_method_QR3(const vector<vector<T>>& matrix, const T& eps, const int& maxIterations) {
    MyResult4<T> result;
    result.iterations = maxIterations;

    int n = matrix.size();
    vector<vector<T>> A(matrix);
    //vector<vector<T>> E = create_identity_matrix<T>(n);
    T max_elem = eps;
    int iter = 0;
    vector<vector<T>> A_result = A;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < maxIterations; j++) {
            max_elem = 0; // Обнуление максимального по модулю элемента текущей строки слева A[i][i]
            vector<vector<T>> Q(A.size(), vector<T>(A.size(), 0)), R(A.size(), vector<T>(A.size(), 0));
            iter += 1;
            T sigma = A[A.size() - 1][A.size() - 1];

            // Алгоритм:
            A = A - sigma * create_identity_matrix<T>(A.size());        // Сдвиг
            //QR_decomposition_with_crop(A, Q, R, eps);                    // QR-разложение

            QR_decomposition(A, Q, R, eps);
            A = R * Q;
            // Обновление A
            A = A + sigma * create_identity_matrix<T>(A.size());        // Обратный сдвиг
            print(A);
            // Ищем максимальный по модулю элемент в строке слева A[i][i]
            for (int j = 0; j < i; ++j) {
                if (abs(A[i][j]) > max_elem)
                    max_elem = abs(A[i][j]);
            }
            // Критерий останова максимальный по модулю элемент текущей строки слева A[i][i] меньше точности
            if (max_elem < eps) {
                break;
            }
        }
        A_result = uncrop(A_result, A, i);
        result.A_iter.push_back(A_result);

        A = crop_matrix(A, 1);
    }

    // Вектор собственных чисел
    vector<T> eigen(n, 0);
    for (int i = 0; i < n; i++) {
        eigen[i] = A_result[i][i];
    }

    result.eigen = sorted(eigen);
    result.R = A_result;
    result.iterations = iter;
    return result;
}

template <typename T>
MyResult4<T> Eigen_method_QR3(const vector<vector<T>>& matrix, const T& sigma, const T& eps, const int& maxIterations) {
    MyResult4<T> result;
    result.iterations = maxIterations;

    int n = matrix.size();
    vector<vector<T>> A(matrix);
    //vector<vector<T>> E = create_identity_matrix<T>(n);
    T max_elem = eps;
    int iter = 0;
    vector<vector<T>> A_result = A;

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j < maxIterations; j++) {
            max_elem = 0; // Обнуление максимального по модулю элемента текущей строки слева A[i][i]
            vector<vector<T>> Q(A.size(), vector<T>(A.size(), 0)), R(A.size(), vector<T>(A.size(), 0));
            iter += 1;
            //T sigma = A[A.size()-1][A.size()-1];

            // Алгоритм:
            A = A - sigma * create_identity_matrix<T>(A.size());        // Сдвиг
            QR_decomposition(A, Q, R, eps);                   // QR-разложение
            A = R * Q;
            // Обновление A
            A = A + sigma * create_identity_matrix<T>(A.size());        // Обратный сдвиг
            print(A);
            // Ищем максимальный по модулю элемент в строке слева A[i][i]
            for (int j = 0; j < i; ++j) {
                if (abs(A[i][j]) > max_elem)
                    max_elem = abs(A[i][j]);
            }
            // Критерий останова максимальный по модулю элемент текущей строки слева A[i][i] меньше точности
            if (max_elem < eps) {
                break;
            }

        }
        A_result = uncrop(A_result, A, i);
        result.A_iter.push_back(A_result);

        A = crop_matrix(A, 1);
    }

    // Вектор собственных чисел
    vector<T> eigen(n, 0);
    for (int i = 0; i < n; i++) {
        eigen[i] = A_result[i][i];
    }

    result.eigen = sorted(eigen);
    result.R = A_result;
    result.iterations = iter;
    return result;
}

/* Функция нахождения собственных векторов матрицы методом Обратных Итераций */
// Сравнение значений двух двух векторов на приближённое равенство независимо от знака
template <typename T>
bool My_Cmp(const vector<T>& x, const vector<T>& y, const T eps)
{
    bool t = true;

    const int n = x.size();

    if (x[0] * y[0] < 0)
        for (int i = 0; i < n; ++i)
            t *= abs(x[i] + y[i]) < eps;
    else
        for (int i = 0; i < n; ++i)
            t *= abs(x[i] - y[i]) < eps;

    return t;
};

template<typename T>
MyResult4<T> reverse_iteration(const vector<vector<T>>& matrix, const vector<T>& lambda, const T& eps, const int& maxIteration) {
    MyResult4<T> result;
    result.eigen = lambda;

    int n = matrix.size();                                         // Размер матрицы
    vector<vector<T>> A(matrix);                                   // Текущая матрица системы
    vector<vector<T>> E = create_identity_matrix<T>(n);
    vector<vector<T>> V;                                           // Результирующая матрица собственных векторов

    vector<T> x(n, 0);
    vector<T> y(n, 0);

    vector<T> x_old(n, 0); // Вектор-обменник

    T x_norm = 0; // Переменная-обменник

    int iter = 0; // Номер итерации

    for (int i = 0; i < n; ++i) {
        x[i] = 1;

        A = A - lambda[i] * E;

        while (iter < maxIteration) {
            y = method_Gaussa2(A, x, /*numeric_limits<T>::epsilon()*/ eps); // Решаем СЛАУ (A - lambda[i]*E)*y = x

            x_norm = norm(y, 2); // Находим евклидову норму y

            x_old = x;

            x = y; // Обновляем вектор х

            for (int j = 0; j < n; j++) {
                x[j] /= x_norm; // Нормировка
            }
            ++iter; // Увеличиваем число итераций на 1


            // Критерий остановы
//            vector<T> delta = x - x_old;
//            if (norm(delta, 2) <= eps) {
//                break;
//            }

            if ((cos_vec(x, x_old) - 1) < eps) {
                break;
            }



        }

        A = matrix;
        V.push_back(x);
        for (T& w : x)
            w = 0;

        x_old = x;
    }
    result.eigens_vec = V;
    result.iterations = iter;
    return result;
}


// Соотношение Рэлея
template <typename T>
T rayleigh_quotient(const vector<vector<T>>& A, const vector<T>& x) {
    int n = A.size();


    // A * x
    vector<T> Ax(n, 0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Ax[i] += A[i][j] * x[j];
        }
    }

    // x^T * A * x
    T xTAx = 0;
    for (int i = 0; i < n; ++i) {
        xTAx += x[i] * Ax[i];
    }

    // x^T * x
    T xTx = 0;
    for (int i = 0; i < n; ++i) {
        xTx += x[i] * x[i];
    }

    if (xTx == 0.0) {
        return 0.0;
    }

    return xTAx / xTx;
}

template <typename T>
MyResult4<T> reverse_iterator_with_reley(const vector<vector<T>>& matrix, const vector<vector<T>>& X0, const T eps, const int& maxIteration) {

    MyResult4<T> result;
    int n = matrix.size();

    vector<vector<T>> A(matrix);
    vector<T> eigens(n, 0);
    vector<vector<T>> vec_lamda(n, vector<T>(n, 0));
    int iter = 0;
    vector<T> x(n, 0);
    vector<T> x_old;


    for (int i = 0; i < n; ++i) {
        x = X0[i];

        for (int j = 0; j < maxIteration; j++) {
            // Решаем (A - lamda *E) * y = x
            vector<T> y = method_Gaussa2(A - eigens[i] * create_identity_matrix<T>(n), x, /*numeric_limits<T>::epsilon() */ eps);
            // Обновляем х
            x_old = x;
            x = y;

            // Нормализаем х
            T y_norm = norm(y, 2);
            for (T& w : x)
                w /= y_norm;

            iter += 1;
            // Проверяем сходимость
            vector<T> delta = x - x_old;
            if (norm(delta, 2) <= eps) {
                break;
            };
            //            if ((cos_vec(x, x_old) - 1) < eps) {
            //                break;
            //            }


        }
        // Обновяем А
        A = matrix;

        // Считаем отношение Рэлея
        eigens[i] = rayleigh_quotient(A, x);

        vec_lamda[i] = x;
    }

    result.eigen = eigens;
    result.eigens_vec = vec_lamda;
    result.iterations = iter;

    return result;
}
