//
//  ActivationFunctions.cpp
//  LearningCPLUSPLUSCUDA
//
//  Created by Tyrell To on 11/12/23.
//

#include "ActivationFunctions.hpp"

Matrix ReLU(const Matrix& matrix) {
    
    Matrix result(matrix.getRows(), matrix.getCols());
    
    for (size_t i =0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j){
            result(i, j) = std::max(0.0, matrix(i, j));
        }
    }
    return result;
}

Matrix LeakyReLU(const Matrix& matrix, double alpha) {
    
    Matrix result(matrix.getRows(), matrix.getCols());
    
    for (size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            double value = matrix(i, j);
            result(i, j) = (value > 0) ? value : alpha * value;
        }
    }
    return result;
}

Matrix Sigmoid(const Matrix& matrix) {
    
    Matrix result(matrix.getRows(), matrix.getCols());
    
    for ( size_t i = 0; i < matrix.getRows(); ++i) {
        for (size_t j = 0; j < matrix.getCols(); ++j) {
            result(i,j) = 1.0 / (1.0 + std::exp(-matrix(i,j)));
        }
    }
    return result;
}

Matrix Softmax(const Matrix& matrix) {
    
    Matrix result(matrix.getRows(), matrix.getCols());
    
    for (size_t j = 0; j < matrix.getCols(); ++j) {
        // Compute the maximum value in the column for numerical stability
        double maxVal = matrix(0, j);
        
        for (size_t i = 1; i < matrix.getRows(); ++i) {
            if (matrix(i, j) > maxVal) {
                maxVal = matrix(i, j);
            }
        }
        
        // Calculate the sum of the exponentials
        double sum = 0.0;
        
        for (size_t i = 0; i < matrix.getRows(); ++i) {
            sum += std::exp(matrix(i, j) - maxVal);
        }
        
        // Calculate softmax
        for (size_t i = 0; i < matrix.getRows(); ++i) {
            result(i, j) = std::exp(matrix(i, j) - maxVal) / sum;
        }
    }
    
    return result;
}

