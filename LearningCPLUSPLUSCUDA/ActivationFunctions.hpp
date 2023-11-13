//
//  ActivationFunctions.hpp
//  LearningCPLUSPLUSCUDA
//
//  Created by Tyrell To on 11/12/23.
//

#ifndef ActivationFunctions_hpp
#define ActivationFunctions_hpp

#include "Matrix.hpp"

Matrix ReLU(const Matrix& matrix);

Matrix LeakyReLU(const Matrix& matrix, double alpha);

Matrix Sigmoid(const Matrix& matrix);

Matrix Softmax(const Matrix& matrix);

#endif /* ActivationFunctions_hpp */
