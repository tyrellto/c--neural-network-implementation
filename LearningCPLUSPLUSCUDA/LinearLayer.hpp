//
//  LinearLayer.hpp
//  LearningCPLUSPLUSCUDA
//
//  Created by Tyrell To on 11/12/23.
//

#ifndef LINEARLAYER_H
#define LINEARLAYER_H

#include "Matrix.hpp"

class LinearLayer {
private:
    Matrix weights; // Matrix holding the weights
    Matrix biases;  // Vector holding the biases
    mutable Matrix inputs;
    Matrix weightsGradient;
    Matrix biasesGradient;

public:
    // Constructor initializes weights and biases
    LinearLayer(size_t numInput, size_t numOutput);

    // Forward pass computes the outputs for a given input
    Matrix forward(const Matrix& inputBatch);

    // (Optional) Functions for backpropagation, if needed
    Matrix backward(const Matrix& outputGradient);
    void updateWeights(double learningRate);
    void updateBiases(double learningRate);

    // Accessor methods
    Matrix getWeights() const { return weights; }
    Matrix getBiases() const { return biases; }
};

#endif // LINEARLAYER_H
