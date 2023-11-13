//
//  LinearLayer.cpp
//  LearningCPLUSPLUSCUDA
//
//  Created by Tyrell To on 11/12/23.
//

#include "LinearLayer.hpp"
#include <vector>
#include <random>
#include <iostream>

// Constructor that initializes weights and biases with small random values
LinearLayer::LinearLayer(size_t numInput, size_t numOutput)
    : weights(numOutput, numInput), biases(numOutput, 1),
      weightsGradient(numOutput, numInput), biasesGradient(numOutput, 1) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, 0.1);

    for (size_t i = 0; i < numOutput; ++i) {
        for (size_t j = 0; j < numInput; ++j) {
            weights(i, j) = dis(gen);
        }
        biases(i, 0) = dis(gen);
    }
}

// Forward pass implementation

Matrix LinearLayer::forward(const Matrix& inputBatch) {
    inputs = inputBatch; // Store the inputs for use in the backward pass
    Matrix outputBatch = weights * inputBatch; // Perform the matrix multiplication Wx + b
    
    for (size_t i = 0; i < outputBatch.getRows(); ++i) {
        for (size_t j = 0; j < outputBatch.getCols(); ++j) {
            outputBatch(i,j) += biases(i,0);
        }
    }
    return outputBatch;
}

Matrix LinearLayer::backward(const Matrix& outputGradient) {
    // Assume that outputGradient is already the average gradient per example in the batch

    // Compute gradient w.r.t. weights (dL/dW = outputGradient * input^T)
    Matrix gradWeights = outputGradient * inputs.transpose();

    // Compute gradient w.r.t. biases (dL/dB = sum of outputGradients over the batch)
    Matrix gradBiases(outputGradient.getRows(), 1); // Initialize bias gradients
    for (size_t i = 0; i < outputGradient.getRows(); ++i) {
        for (size_t j = 0; j < outputGradient.getCols(); ++j) {
            gradBiases(i, 0) += outputGradient(i, j);
        }
    }

    // Compute gradient w.r.t. inputs (dL/dInput = W^T * outputGradient)
    Matrix inputGradient = weights.transpose() * outputGradient;

    // Store the gradients
    weightsGradient = gradWeights;
    biasesGradient = gradBiases;

    return inputGradient;
}

void LinearLayer::updateWeights(double learningRate) {
    // Assuming weightsGradient is computed and stored in the backward pass
    // Update the weights in the opposite direction of the gradient
    weights = weights - (weightsGradient * learningRate);
}

void LinearLayer::updateBiases(double learningRate) {
    // Assuming biasesGradient is computed and stored in the backward pass
    // Update the biases in the opposite direction of the gradient
    biases = biases - (biasesGradient * learningRate);
}

