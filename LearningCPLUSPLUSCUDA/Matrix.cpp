//
//  Matrix.cpp
//  LearningCPLUSPLUSCUDA
//
//  Created by Tyrell To on 11/12/23.
//

#include "Matrix.hpp"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <iomanip>


Matrix::Matrix(size_t rows, size_t cols) : rows(rows), cols(cols), data(rows, std::vector<double>(cols, 0.0)) {}

Matrix::~Matrix() {}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        rows = other.rows;
        cols = other.cols;
        data = other.data;
    }
    return *this;
}

// Access individual elements only
double& Matrix::operator()(const size_t& row, const size_t& col) {
    return data[row][col];
}

const double& Matrix::operator()(const size_t& row, const size_t& col) const {
    return data[row][col];
}

size_t Matrix::getRows() const{
    return rows;
}

size_t Matrix::getCols() const{
    return cols;
}

Matrix Matrix::operator+(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices have different dimensions!");
    }
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i,j) = data[i][j] + other(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator-(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices have different dimensions!");
    }
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i,j) = data[i][j] - other(i,j);
        }
    }
    return result;
}

Matrix Matrix::operator*(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Matrices have different dimensions!");
    }
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                result(i,j) += data[i][k] * other(k,j);
            }
        }
    }
    return result;
}

Matrix Matrix::operator*(double scalar) const{
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i,j) = data[i][j] * scalar;
        }
    }
    return result;
}

Matrix Matrix::elementWiseMultiplication(const Matrix& other) const {
    if (rows != other.rows || cols != other.cols){
        throw std::invalid_argument("Matrices have different dimensions!");
    }
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i,j) = data[i][j] * other(i,j);
        }
    }
    return result;
}


Matrix Matrix::transpose() const {
    
    Matrix result(rows, cols);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(j,i) = data[i][j];
        }
    }
    return result;
}

Matrix Matrix::forwardSubstitution(const Matrix& L, const Matrix& b) const {
    if (L.rows != b.rows) {
        throw std::invalid_argument("Incompatible dimensions for forward substitution.");
    }

    Matrix y(L.rows, 1);
    for (size_t i = 0; i < L.rows; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < i; ++j) {
            sum += L(i, j) * y(j, 0);
        }
        y(i, 0) = (b(i, 0) - sum) / L(i, i);
    }
    return y;
}

Matrix Matrix::backwardSubstitution(const Matrix& U, const Matrix& y) const {
    if (U.rows != y.rows) {
        throw std::invalid_argument("Incompatible dimensions for backward substitution.");
    }

    Matrix x(U.rows, 1);
    for (size_t i = U.rows - 1; i >= 0; --i) {
        double sum = 0.0;
        for (size_t j = i + 1; j < U.cols; ++j) {
            sum += U(i, j) * x(j, 0);
        }
        x(i, 0) = (y(i, 0) - sum) / U(i, i);
    }
    return x;
}

void Matrix::luDecomposition(Matrix& L, Matrix& U) const {
    if (rows != cols) {
        throw std::invalid_argument("LU decomposition not defined for non-square matrices.");
    }

    L = Matrix(rows, cols);
    U = Matrix(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        // Upper Triangular
        for (size_t k = i; k < rows; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < i; ++j) {
                sum += (L(i, j) * U(j, k));
            }
            U(i, k) = data[i][k] - sum;
        }

        // Lower Triangular
        for (size_t k = i; k < rows; ++k) {
            if (i == k) {
                L(i, i) = 1; // Diagonal as 1
            } else {
                double sum = 0.0;
                for (size_t j = 0; j < i; ++j) {
                    sum += (L(k, j) * U(j, i));
                }
                L(k, i) = (data[k][i] - sum) / U(i, i);
            }
        }
    }
}

double Matrix::determinant() const {
    // Ensure the matrix is square
    if (rows != cols) {
        throw std::invalid_argument("Determinant is not defined for non-square matrices.");
    }

    // Decompose the matrix into L and U components
    Matrix L(rows, cols), U(rows, cols);
    luDecomposition(L, U);

    // The determinant is the product of the diagonal elements of U
    double det = 1.0;
    
    for (size_t i = 0; i < rows; ++i) {
        det *= U(i, i);
    }
    
    return det;
}


Matrix Matrix::inverse() const {
    double det = this->determinant();
    // Use a tolerance for comparing floating-point numbers
    const double tolerance = 1e-9;
    
    if (std::abs(det) < tolerance) {
        throw std::invalid_argument("Matrix is singular and cannot be inverted.");
    }

    // Perform LU decomposition on the matrix
    Matrix L(rows, cols), U(rows, cols);
    this->luDecomposition(L, U);

    // Initialize the inverse matrix
    Matrix inverseMat(rows, cols);

    // Initialize identity matrix once
    Matrix identity(rows, 1);
    for (size_t k = 0; k < rows; ++k) {
        identity(k, 0) = 1.0;
    }

    // Solve for each column of the inverse matrix
    for (size_t i = 0; i < rows; ++i) {
        
        // Use column 'i' of the identity matrix
        Matrix e(rows, 1);
        for (size_t j = 0; j < rows; ++j) {
            e(j, 0) = (i == j) ? 1.0 : 0.0;
        }
        
        // Solve Ly = e for y
        Matrix y = forwardSubstitution(L, e);

        // Solve Ux = y for x, which will be the i-th column of the inverse
        Matrix x = backwardSubstitution(U, y);

        // Set the i-th column of the inverse matrix
        for (size_t j = 0; j < rows; ++j) {
            inverseMat(j, i) = x(j, 0);
        }
    }
    return inverseMat;
}

double Matrix::frobeniusNorm() const {
    double sum = 0.0;
    for (size_t i =0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum += data[i][j] * data[i][j];
        }
    }
    return std::sqrt(sum);
}

double Matrix::dotProduct(const Matrix& other) const {
    if ((rows != 1 && cols != 1) || (other.rows != 1 && other.cols != 1)) {
        throw std::invalid_argument("One of the matrices must be a row vector and the other a column vector.");
    }
    // Check that the number of elements is the same
    if (rows * cols != other.rows * other.cols) {
        throw std::invalid_argument("The matrices must have the same number of elements.");
    }
    
    double product = 0.0;
    
    if (rows == 1) {
        for (size_t i = 0; i < cols; ++i) {
            product += data[0][i] * other.data[i][0];
        }
    }
    else {
        for (size_t i =0; i < rows; ++i) {
            product += data[i][0] * other.data[0][i];
        }
    }
    
    return product;
}

// Function to get the diagonal of the matrix
std::vector<double> Matrix::diagonal() const {
    size_t minDim = std::min(rows, cols);
    std::vector<double> diagonal(minDim);
    
    for (size_t i = 0; i < minDim; ++i) {
        diagonal[i] = data[i][i];
    }
    
    return diagonal;
}


double Matrix::trace() const {
    if (rows != cols) {
        throw std::invalid_argument("Trace is not defined for non-square matrices.");
    }
    
    double trace = 0.0;
    
    for (size_t i = 0; i < rows; ++i) {
        trace += data[i][i];
    }
    
    return trace;
}

// Function for horizontal concatenation
Matrix Matrix::concatenateHorizontally(const Matrix& rhs) const {
    if (rows != rhs.rows) {
        throw std::invalid_argument("Row counts must match for horizontal concatenation.");
    }

    // The new matrix will have a column count that is the sum of the two matrices' column counts
    Matrix result(rows, cols + rhs.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j];
        }
        for (size_t j = 0; j < rhs.cols; ++j) {
            result(i, cols + j) = rhs(i, j);
        }
    }
    return result;
}

// Function for vertical concatenation
Matrix Matrix::concatenateVertically(const Matrix& rhs) const {
    if (cols != rhs.cols) {
        throw std::invalid_argument("Column counts must match for vertical concatenation.");
    }

    // The new matrix will have a row count that is the sum of the two matrices' row counts
    Matrix result(rows + rhs.rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(i, j) = data[i][j];
        }
    }
    for (size_t i = 0; i < rhs.rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result(rows + i, j) = rhs(i, j);
        }
    }
    return result;
}

Matrix Matrix::broadcast(size_t newRows, size_t newCols) const {
    if (rows != 1 && rows != newRows) {
        throw std::invalid_argument("Broadcasting is not possible along the row dimension.");
    }
    if (cols != 1 && cols != newCols) {
        throw std::invalid_argument("Broadcasting is not possible along the column dimension.");
    }

    Matrix result(newRows, newCols);
    for (size_t i = 0; i < newRows; ++i) {
        for (size_t j = 0; j < newCols; ++j) {
            // Copy the element from the original matrix, replicating as necessary
            size_t originalRow = (rows == 1) ? 0 : i;
            size_t originalCol = (cols == 1) ? 0 : j;
            result(i, j) = data[originalRow][originalCol];
        }
    }
    return result;
}

void Matrix::reshape(size_t newRows, size_t newCols) {
    // Check if the total number of elements is the same
    if (newRows * newCols != rows * cols) {
        throw std::invalid_argument("New dimensions must contain the same number of elements");
    }

    // Flatten the matrix into a temporary 1D vector
    std::vector<double> temp;
    temp.reserve(rows * cols);
    for (const auto& row : data) {
        for (const auto& elem : row) {
            temp.push_back(elem);
        }
    }

    // Clear the current matrix data and resize it to the new dimensions
    data.clear();
    data.resize(newRows, std::vector<double>(newCols));

    // Copy the elements from the temporary vector into the resized matrix
    for (size_t i = 0; i < newRows; ++i) {
        for (size_t j = 0; j < newCols; ++j) {
            data[i][j] = temp[i * newCols + j];
        }
    }

    // Update the matrix dimensions
    rows = newRows;
    cols = newCols;
}

void Matrix::print() const {
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << data[i][j] << " ";
        }
        std::cout << std::endl; // End of a matrix row
    }
}


//Activation Function: Implement the ReLU activation function and its derivative for backpropagation.
//
//Loss Function: Implement a loss function like Mean Squared Error or Cross-Entropy, along with a function to compute its gradient.
//
//Layers: Define classes for different types of layers (e.g., Dense/Fully Connected). Each layer should be able to perform a forward and backward pass.
//
//Network Class: Define a neural network class that combines the layers and implements the forward and backward pass for the whole network.
//
//Optimizer: Implement gradient descent optimization to adjust the weights of the network.
//
//Training Loop: Create a training loop that feeds input data to the network and uses the optimizer to update the weights based on the loss gradient.
//Matrix-Norm
//A commonly used matrix norm in machine learning is the Frobenius norm, which is the square root of the sum of the absolute squares of its elements.
//
//Dot Product
//The dot product of two vectors represented as matrices can be thought of as a special case of matrix multiplication where both matrices have either a single row or column.
//
//Diagonal
//For a matrix class, getting the diagonal might involve iterating over the minimum of the number of rows or columns and accessing the i, i element.
//
//Trace
//The trace of a matrix is the sum of the elements on the main diagonal, which can be calculated in a similar way to extracting the diagonal.
//
//Row Reduction/Echelon Form
//This is part of Gaussian elimination. You perform row operations to transform the matrix into a row-echelon form where all elements below the main diagonal are zero.
//
//Concatenation
//Concatenating two matrices either by rows or columns involves creating a new matrix with dimensions that are the sum of the original matrices in the direction of concatenation and then filling in the elements appropriately.
//
//Reshape
//Reshaping a matrix is changing its dimensions while keeping the same elements. It's important to ensure that the total number of elements remains constant.
//
//Broadcasting
//This is the process of making matrices with different shapes have compatible shapes for arithmetic operations. The smaller matrix is "broadcast" across the larger one by replicating the smaller matrix along the necessary dimensions.
