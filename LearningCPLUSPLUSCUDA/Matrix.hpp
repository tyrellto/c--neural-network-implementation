// Matrix.h
#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <cstddef> // for size_t


class Matrix {
private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

public:
    // Constructors, destructors, and assignment operators as needed
    Matrix() : rows(0), cols(0), data() {}  // Default constructor
    Matrix(size_t rows, size_t cols); // Assume this is already implemented
    Matrix(const Matrix& other);      // Copy constructor
    ~Matrix();                        // Destructor
    Matrix& operator=(const Matrix& other); // Assignment operator

    // Access elements
    double& operator()(const size_t& row, const size_t& col);
    const double& operator()(const size_t& row, const size_t& col) const;

    // Get dimensions
    size_t getRows() const;
    size_t getCols() const;

    // Matrix operations
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;
    Matrix operator*(double scalar) const;
    Matrix elementWiseMultiplication(const Matrix& other) const;

    // Transposition
    Matrix transpose() const;

    // Substitution and decomposition methods
    Matrix forwardSubstitution(const Matrix& L, const Matrix& b) const;
    Matrix backwardSubstitution(const Matrix& U, const Matrix& y) const;
    void luDecomposition(Matrix& L, Matrix& U) const;

    // Determinant and inverse
    double determinant() const;
    Matrix inverse() const;

    // Norms and other properties
    double frobeniusNorm() const;
    double dotProduct(const Matrix& other) const;
    std::vector<double> diagonal() const;
    double trace() const;

    // Concatenation
    Matrix concatenateHorizontally(const Matrix& rhs) const;
    Matrix concatenateVertically(const Matrix& rhs) const;

    // Broadcasting and reshaping
    Matrix broadcast(size_t newRows, size_t newCols) const;
    void reshape(size_t newRows, size_t newCols);

    // Utility functions
    void print() const; // Assuming you'd also want a print method for convenience

    // ... any additional member functions ...
};

#endif // MATRIX_H
