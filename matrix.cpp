#include <iostream>
#include <vector>
#include <math.h>
#include "matrix.hpp"
#include <random>

using namespace std;

std::random_device rd;
std::mt19937 gen(rd());

#define DEFAULT_MATRIX_VALUE 0

matrix::matrix()
{
    matrix_body = nullptr;
    dimensions = {};
    size = 0;
}

matrix::matrix(vector<int> dimensions)
{
    int size = 1;
    for (int i = 0; i < (int)dimensions.size(); i++)
    {
        size *= dimensions[i];
    }
    matrix_body = (float *)malloc(size * sizeof(float));

    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = DEFAULT_MATRIX_VALUE;
    }

    this->matrix_body = matrix_body;
    this->dimensions = dimensions;
    this->size = size;
}

matrix::matrix(vector<int> dimensions, bool rand) : matrix(dimensions)
{
    if (rand)
    {
        urand();
        // nrand();
    }
}

matrix::~matrix()
{
    free(matrix_body);
}

void matrix::print_true()
{
    for (int i = 0; i < size; i++)
    {
        cout << matrix_body[i] << " ";
    }
    cout << endl;
}

void matrix::print()
{
    vector<int> idx = {0, 0};
    for (int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < dimensions[1]; j++)
        {
            idx = {i, j};
            printf("%f ", matrix_body[get_true_index(idx)]);
        }
        printf("\n");
    }
}

void matrix::urand()
{
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = ((float)rand()) / (float)(RAND_MAX) * 2 - 1;
    }
}

void matrix::nrand()
{
    for (int i = 0; i < size; i++)
    {
        normal_distribution<float> d(0, 0.5);
        matrix_body[i] = d(gen);
    }
}

matrix matrix::add(matrix &input)
{
    if (dimensions != input.dimensions)
    {
        cout << "Error: Matrix Addition Error" << endl;
    }
    matrix res = matrix(dimensions);
    for (int i = 0; i < size; i++)
    {
        res.matrix_body[i] = matrix_body[i] + input.matrix_body[i];
    }
    return res;
}

void matrix::add_inplace(matrix &input)
{
    if (dimensions != input.dimensions)
    {
        cout << "Error: Matrix Addition Error" << endl;
    }
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = matrix_body[i] + input.matrix_body[i];
    }
}

matrix matrix::mult(matrix &input)
{

    // When multiplying an n x m matrix by m x q matrix, we get a resulting n x q matrix
    if (this->dimensions[1] != input.dimensions[0])
    {
        cout << "Error: Matrix Multiplication Error" << endl;
    }

    vector<int> dimensions = {this->dimensions[0], input.dimensions[1]};
    matrix res = matrix(dimensions);

    // Using matrix multiplication formula: C_ij = sum_k A_ikB_kj
    float sum;
    vector<int> idx1 = {0, 0};
    vector<int> idx2 = {0, 0};
    vector<int> idx3 = {0, 0};
    for (int res_row = 0; res_row < res.dimensions[0]; res_row++)
    {
        for (int res_col = 0; res_col < res.dimensions[1]; res_col++)
        {
            sum = 0;
            for (int m1_col = 0; m1_col < input.dimensions[0]; m1_col++)
            {
                idx1 = {res_row, m1_col};
                idx2 = {m1_col, res_col};
                sum += (matrix_body[get_true_index(idx1)]) *
                       (input.matrix_body[input.get_true_index(idx2)]);
            }
            idx3 = {res_row, res_col};
            res.matrix_body[res.get_true_index(idx3)] = sum;
        }
    }

    return res;
}

// mult_inplace(matrix &B, matrix &A, output_matrix &A)

void matrix::mult_inplace(matrix &input, bool M_A = true)
{

    if (this->dimensions[1] != input.dimensions[0])
    {
        cout << "Error: Matrix Multiplication Error" << endl;
        return;
    }

    vector<int> new_dimensions = {this->dimensions[0], input.dimensions[1]};

    int new_size = new_dimensions[0] * new_dimensions[1];
    float *new_matrix_body = (float *)malloc(new_size * sizeof(float));

    float sum;
    vector<int> idx1 = {0, 0};
    vector<int> idx2 = {0, 0};
    int idx3;
    for (int res_row = 0; res_row < new_dimensions[0]; res_row++)
    {
        for (int res_col = 0; res_col < new_dimensions[1]; res_col++)
        {
            sum = 0;
            for (int m1_col = 0; m1_col < dimensions[1]; m1_col++)
            {
                idx1 = {res_row, m1_col};
                idx2 = {m1_col, res_col};
                sum += (matrix_body[get_true_index(idx1)]) *
                       (input.matrix_body[input.get_true_index(idx2)]);
            }
            idx3 = res_row * new_dimensions[1] + res_col;
            new_matrix_body[idx3] = sum;
        }
    }

    float *old_matrix_body = input.matrix_body;

        free(old_matrix_body);
    old_matrix_body = nullptr;
}

// ollie's multithreading
/* matrix matrix::mult(matrix &input) {
    // When multiplying an n x m matrix by m x q matrix, we get a resulting n x q matrix
    if (this->dimensions[1] != input.dimensions[0]) {
        cout << "Error: Matrix Multiplication Error" << endl;
        // Handle the error appropriately
    }

    vector<int> dimensions = {this->dimensions[0], input.dimensions[1]};
    matrix res = matrix(dimensions);

    // Calculate the number of threads based on the dimensions of the result matrix
    int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;

    // Split the work among threads
    int rows_per_thread = res.dimensions[0] / num_threads;
    for (int t = 0; t < num_threads; t++) {
        threads.emplace_back([&](int thread_id) {
            int start_row = thread_id * rows_per_thread;
            int end_row = (thread_id == num_threads - 1) ? res.dimensions[0] : start_row + rows_per_thread;

            for (int res_row = start_row; res_row < end_row; res_row++) {
                for (int res_col = 0; res_col < res.dimensions[1]; res_col++) {
                    float sum = 0;
                    for (int m1_col = 0; m1_col < input.dimensions[0]; m1_col++) {
                        sum += (matrix_body[get_true_index({res_row, m1_col})]) *
                               (input.matrix_body[input.get_true_index({m1_col, res_col})]);
                    }
                    res.matrix_body[res.get_true_index({res_row, res_col})] = sum;
                }
            }
        }, t);
    }

    // Wait for all threads to finish
    for (auto &thread : threads) {
        thread.join();
    }

    return res;
}
*/

int matrix::get_true_index(vector<int> &indices)
{
    int index = 0;
    int size_copy = size;
    for (int i = 0; i < (int)indices.size(); i++)
    {
        size_copy /= dimensions[i];
        index += indices[i] * size_copy;
    }
    return index;
}

vector<int> matrix::get_psuedo_indices(int true_index)
{

    if (true_index < 0 || true_index >= size)
    {
        cout << "Error: true index out of range: in  get_psuedo_index" << endl;
    }

    vector<int> pseudo_indices(dimensions.size());

    int temp_size = size;
    int remaining_index = true_index;

    for (int i = 0; i < (int)dimensions.size() - 1; i++)
    {
        temp_size /= dimensions[i];
        pseudo_indices[i] = remaining_index / temp_size;
        remaining_index %= temp_size;
    }

    pseudo_indices[dimensions.size() - 1] = remaining_index;

    return pseudo_indices;
}

void matrix::set_value(vector<int> &indices, float value)
{
    int index = get_true_index(indices);
    matrix_body[index] = value;
}

void matrix::set_value(int index, float value)
{
    matrix_body[index] = value;
}

void matrix::print_shape()
{
    for (int i = 0; i < (int)dimensions.size(); i++)
    {
        cout << dimensions[i] << " ";
    }
    cout << endl;
}

void matrix::copy(matrix &input)
{
    dimensions = input.dimensions;
    size = input.size;
    matrix_body = (float *)realloc(matrix_body, size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = input.matrix_body[i];
    }
}

int matrix::get_max_index()
{
    int max_index = 0;
    float max_value = matrix_body[0];
    for (int i = 1; i < size; i++)
    {
        if (matrix_body[i] > max_value)
        {
            max_value = matrix_body[i];
            max_index = i;
        }
    }
    return max_index;
}

void matrix::transpose_inplace()
{
    vector<int> idx1 = {dimensions[1], dimensions[0]};
    vector<int> idx2 = {0, 0};
    matrix transposed(idx1);

    for (int i = 0; i < dimensions[0]; i++)
    {
        for (int j = 0; j < dimensions[1]; j++)
        {
            idx1 = {i, j};
            idx2 = {j, i};
            int original_index = this->get_true_index(idx1);
            int transposed_index = transposed.get_true_index(idx2);
            transposed.matrix_body[transposed_index] = matrix_body[original_index];
        }
    }

    dimensions = transposed.dimensions;
    size = transposed.size;
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = transposed.matrix_body[i];
    }
}

void matrix::hammard_product_inplace(matrix &input)
{
    if (dimensions != input.dimensions)
    {
        cout << "Error: Hammard Product Error" << endl;
        return;
    }

    for (int i = 0; i < input.size; i++)
    {
        matrix_body[i] = matrix_body[i] * input.matrix_body[i];
    }
}

void matrix::scalar_mult_inplace(float scalar)
{
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] *= scalar;
    }
}

float matrix::true_dot(matrix A, matrix B)
{
    if (A.size != B.size)
    {
        cout << "Error: True Dot Error" << endl;
    }

    float sum = 0;
    for (int i = 0; i < A.size; i++)
    {
        sum += A.matrix_body[i] * B.matrix_body[i];
    }

    return sum;
}

void matrix::sub_inplace(matrix &input)
{
    if (dimensions != input.dimensions)
    {
        cout << "Error: Matrix Subtraction Error" << endl;
    }
    for (int i = 0; i < size; i++)
    {
        matrix_body[i] = matrix_body[i] - input.matrix_body[i];
    }
}