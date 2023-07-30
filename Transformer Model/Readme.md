# Transformer Neural Network Project

This Is a project that Implements a **Transformer neural network** Using Visual Basic (VB). The Transformer architecture was introduced In the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. And has been widely used in natural language processing tasks, such as machine translation, language modeling, And text generation.

## Overview

The project consists Of several classes that together create a functional Transformer neural network. The main components Of the Transformer are implemented, including **self-attention mechanism**, **feed-forward network**, **layer normalization**, And **cross-entropy loss**. The neural network Is designed To be modular And flexible, allowing easy customization And extension.

## Classes

1. **`MathMult`**: A utility Class containing functions For matrix operations, such As dot product, transposition, softmax, And concatenation.

2. **`LinearLayer`**: A Class representing a linear layer used In the Transformer. It performs linear transformations (matrix multiplications) On input sequences.

3. **`ScaledDotProductAttention`**: A Class implementing the scaled dot-product attention mechanism, which Is the core part Of the Transformer's attention mechanism.

4. **`LayerNormalization`**: A Class implementing layer normalization, which normalizes the inputs within Each sequence.

5. **`FeedForwardNetwork`**: A Class representing the feed-forward network used In the Transformer.

6. **`CrossEntropyLoss`**: A Class for computing the cross-entropy loss And its gradients.

7. **`Transformer`**: The main Class that assembles all the components Of the Transformer neural network, including multi-headed self-attention And feed-forward blocks.

## Usage

The project Is designed To be used For tasks that benefit from Transformer-based models. For specific tasks, follow these steps

1. Create Or load your dataset And preprocess it into sequences of numerical features.

2. Initialize the `Transformer` class with appropriate parameters, such as the number of attention heads, hidden layer sizes, And the size of the input And output sequences.

3. Use the `Transformer` class to train the model on your dataset using the appropriate training algorithm (e.g., gradient descent, Adam optimizer, etc.).

4. Once the model Is trained, use it to make predictions on New data Or evaluate its performance on a test dataset.

## Example Usage

```vb
' Create a Transformer neural network
Dim transformer As New Transformer(inputSize:=100, hiddenSize:=256, numHeads:=8)

                    ' Prepare your dataset and labels
                    Dim trainingData As List(Of List(Of Double)) = LoadTrainingData()
                    Dim labels As List(Of List(Of Double)) = LoadLabels()

' Train the model using gradient descent
transformer.Train(trainingData, labels, learningRate:=0.001, numEpochs:=100)

' Make predictions on new data
Dim newData As List(Of List(Of Double)) = PrepareNewData()
                    Dim predictions As List(Of List(Of Double)) = transformer.Predict(newData)
```

## Requirements

To run this project, you need

- Visual Basic .NET environment
- The `System` namespace

## Contributing

Contributions to this project are welcome. If you find any issues Or have suggestions for improvements, feel free to submit a pull request Or open an issue.

## License

This project Is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Credits

The Transformer neural network implementation In this project Is inspired by the original paper "Attention Is All You Need" by Vaswani et al. The code And Structure Of the neural network were created based On their work.

## Acknowledgments

Thanks to the authors of the Transformer paper And the open-source community for providing valuable resources And tools. We acknowledge the contributions of the research And development community that make such projects possible.