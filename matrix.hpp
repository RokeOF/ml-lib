#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>

using namespace std;

class matrix
{
public:
    // constructors
    matrix();
    matrix(vector<int> dimensions);
    matrix(vector<int> dimensions, bool rand);
    ~matrix();

    // attributes
    vector<int> dimensions;
    float *matrix_body;
    int size;

    // methods
    void print();
    void print_true();
    int get_true_index(vector<int> &indices);
    vector<int> get_psuedo_indices(int true_index);
    void set_value(vector<int> &indices, float value);
    void set_value(int index, float value);
    void urand();
    void nrand();
    void print_shape();
    void copy(matrix &input);
    int get_max_index();
    void transpose_inplace();

    // operations
    matrix add(matrix &input);
    void add_inplace(matrix &input);
    // matrix sub(matrix &input);
    void sub_inplace(matrix &input);
    matrix mult(matrix &input);
    void mult_inplace(matrix &input);
    void scalar_mult_inplace(float scalar);
    float true_dot(matrix A, matrix B);
    void hammard_product_inplace(matrix &input);
};

#endif // MATRIX_HPP
