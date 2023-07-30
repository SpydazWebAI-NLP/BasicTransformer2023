# Basic Transformer Neural Network

This project implements a basic version of the Transformer neural network using Visual Basic (VB). The Transformer architecture was introduced in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. and has been widely used in natural language processing tasks, such as machine translation, language modeling, and text generation.

## Overview

The `BasicTransformer` class is the main component of the Transformer neural network. It consists of several methods that perform different parts of the Transformer's architecture, such as self-attention, feed-forward network, layer normalization, and masking. The `BasicTransformer` class is designed to be a simplified version of the Transformer and can be used for various tasks that benefit from the Transformer's attention mechanism.

## Usage

To use the `BasicTransformer` class, follow these steps:

1. Create or load your vocabulary and preprocess it into numerical representations.

2. Initialize the `BasicTransformer` class with the appropriate `embeddingSize` and `vocabulary`.

3. Use the `EncodeQuery` method to encode the input data.

4. Use the `Train` method to train the model on your dataset using the appropriate training algorithm (e.g., gradient descent, Adam optimizer, etc.).

5. Use the `Forward` method to make predictions on new data.

## Example Usage

```vb
' Load vocabulary and prepare data
Dim vocabulary As New List(Of String)() From {"apple", "banana", "orange", ...}
Dim data As List(Of List(Of Double)) = PrepareData()

' Create the basic Transformer model
Dim transformer As New BasicTransformer(embeddingSize:=100, vocabulary)

' Encode the input data
Dim encodedData As List(Of List(Of Double)) = transformer.EncodeQuery(data)

' Train the model on the encoded data and target data
Dim targetData As List(Of List(Of Double)) = PrepareTargetData()
Dim predictions As List(Of List(Of Double)) = transformer.Train(encodedData, targetData)

' Make predictions on new data
Dim newData As List(Of List(Of Double)) = PrepareNewData()
Dim newEncodedData As List(Of List(Of Double)) = transformer.EncodeQuery(newData)
Dim newPredictions As List(Of List(Of Double)) = transformer.Forward(newEncodedData)
```

## Requirements

To run this project, you need:

- Visual Basic .NET environment
- The `System` namespace

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, feel free to submit a pull request or open an issue.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

## Credits

The `BasicTransformer` class is inspired by the original paper "Attention Is All You Need" by Vaswani et al. The code and structure of the neural network were created based on their work.

## Acknowledgments

Thanks to the authors of the Transformer paper and the open-source community for providing valuable resources and tools. We acknowledge the contributions of the research and development community that make such projects possible.