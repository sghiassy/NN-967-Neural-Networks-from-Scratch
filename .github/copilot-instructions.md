# Copilot Instructions for Neural Networks from Scratch

## Overview
This project is designed to help users learn neural networks from scratch using Python. It includes various chapters that cover fundamental concepts and practical implementations.

## Project Structure
- **Chpt02/**: Contains basic implementations of neural network components, such as weights and dot products.
- **Chpt03/**: Expands on the previous chapter by adding more layers and handling spiral data.

## Key Components
- **Layer_Dense**: A class representing a dense layer in a neural network. It initializes weights and biases and computes the forward pass.

## Developer Workflows
- **Running Scripts**: Each script in the `Chpt02` and `Chpt03` directories can be run independently to test specific functionalities.
- **Testing**: Ensure to validate outputs by checking the printed results in the console.

## Integration Points
- **Data Handling**: The `spiral_data` function from `nnfs.datasets` is used to generate sample data for testing.

## Conventions and Patterns
- **Naming Conventions**: Classes are named using CamelCase (e.g., `Layer_Dense`), while functions and variables use snake_case.
- **Documentation**: Each class and function should be documented with docstrings explaining their purpose and usage.

## Example Usage
To create a dense layer and perform a forward pass:
```python
from nnfs.datasets import spiral_data

X, y = spiral_data(samples=100, classes=3)
dense1 = Layer_Dense(2, 3)
dense1.forward(X)
print(dense1.output[:5])
```

## Conclusion
This document serves as a guide for AI coding agents to understand the structure and workflows of the project. For further details, refer to the individual scripts in the chapters.