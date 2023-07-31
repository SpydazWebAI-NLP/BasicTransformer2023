Imports System.Math

Namespace iTransform
    Public Class TransformerLayer
        Public Property EmbeddingSize As Integer
        Public Property SelfAttentionDim As Integer
        Public Property FeedForwardDim As Integer
        Public Property Epsilon As Double
        Public Property NumHeads As Integer
        Public Property SelfAttentionMask As Double()
        Public Property EncoderDecoderAttentionMask As Double()

        Public Sub New(embeddingSize As Integer, selfAttentionDim As Integer, feedForwardDim As Integer,
                   epsilon As Double, numHeads As Integer)
            Me.EmbeddingSize = embeddingSize
            Me.SelfAttentionDim = selfAttentionDim
            Me.FeedForwardDim = feedForwardDim
            Me.Epsilon = epsilon
            Me.NumHeads = numHeads

            ' Initialize self-attention mask and encoder-decoder attention mask
            ' ... Implement the mask initialization as needed ...
        End Sub
        Private Sub LayerNormalize(ByRef vector As Double())
            ' Compute the mean and variance of the vector.
            Dim mean As Double = vector.Average()
            Dim variance As Double = vector.Select(Function(x) (x - mean) * (x - mean)).Average()

            ' Normalize the vector.
            For i As Integer = 0 To vector.Length - 1
                vector(i) = (vector(i) - mean) / Math.Sqrt(variance + Epsilon)
            Next
        End Sub
        Private Sub AddAndNorm(ByRef decoderOutput As List(Of Double()), ByRef sourceEmbeddings As List(Of Double()))
            ' Add the output of the decoder and the source embeddings.
            For i As Integer = 0 To decoderOutput.Count - 1
                For j As Integer = 0 To decoderOutput(i).Length - 1
                    decoderOutput(i)(j) += sourceEmbeddings(i)(j)
                Next
            Next

            ' Apply layer normalization to the combined output.
            For i As Integer = 0 To decoderOutput.Count - 1
                LayerNormalize(decoderOutput(i))
            Next
        End Sub
    End Class
    Public Class TransformerModel
        ' Transformer model configuration
        Private EncoderLayers As List(Of TransformerLayer)
        Private DecoderLayers As List(Of TransformerLayer)
        Private WordEmbeddings As Embeddings.Factory.WordEmbeddingsModel
        Private EmbeddingSize As Integer
        Private NumEncoderLayers As Integer
        Private NumDecoderLayers As Integer
        Private NumHeads As Integer
        Private SelfAttentionDim As Integer
        Private FeedForwardDim As Integer
        Private Epsilon As Double
        Private Beta1 As Double
        Private Beta2 As Double
        Private LearningRate As Double
        Private NumEpochs As Integer
        Private ReadOnly m_dict As New Dictionary(Of String, Double())
        Private ReadOnly v_dict As New Dictionary(Of String, Double())

        Public Structure TrainingDta
            Dim inputSeq As List(Of List(Of Double))
            Dim targetSeq As List(Of List(Of Double))

        End Structure
        Public Sub Train(trainData As TrainingDta, numEpochs As Integer, learningRate As Double)
            For epoch = 1 To numEpochs
                Dim totalLoss As Double = 0.0

                ' Forward pass
                Dim predictions = Forward(trainData.inputSeq)

                ' Calculate loss
                Dim loss = CrossEntropyLoss.ComputeCrossEntropyLoss(predictions, trainData.targetSeq)
                totalLoss += loss

                ' Backward pass
                Dim gradients = CrossEntropyLoss.ComputeGradients(predictions, trainData.targetSeq)


                Dim averageLoss As Double = totalLoss / trainData.inputSeq.Count
                Console.WriteLine($"Epoch {epoch}, Loss: {averageLoss}")
            Next
        End Sub
        Public Sub New(embeddingSize As Integer, numEncoderLayers As Integer, numDecoderLayers As Integer,
                   numHeads As Integer, selfAttentionDim As Integer, feedForwardDim As Integer,
                   epsilon As Double, beta1 As Double, beta2 As Double, learningRate As Double, numEpochs As Integer,
                   vocabulary As List(Of String))
            Me.EmbeddingSize = embeddingSize
            Me.NumEncoderLayers = numEncoderLayers
            Me.NumDecoderLayers = numDecoderLayers
            Me.NumHeads = numHeads
            Me.SelfAttentionDim = selfAttentionDim
            Me.FeedForwardDim = feedForwardDim
            Me.Epsilon = epsilon
            Me.Beta1 = beta1
            Me.Beta2 = beta2
            Me.LearningRate = learningRate
            Me.NumEpochs = numEpochs

            ' Initialize the word embeddings model
            WordEmbeddings = New Embeddings.Factory.HybridWordEmbeddingsModel(vocabulary)

            ' Initialize encoder and decoder layers
            EncoderLayers = New List(Of TransformerLayer)
            For i As Integer = 0 To numEncoderLayers - 1
                EncoderLayers.Add(New TransformerLayer(embeddingSize, selfAttentionDim, feedForwardDim, epsilon, numHeads))
            Next

            DecoderLayers = New List(Of TransformerLayer)
            For i As Integer = 0 To numDecoderLayers - 1
                DecoderLayers.Add(New TransformerLayer(embeddingSize, selfAttentionDim, feedForwardDim, epsilon, numHeads))
            Next
        End Sub

        ' Layer normalization function
        Private Function LayerNormalize(x As Double()(), epsilon As Double) As Double()()
            Dim numInputs As Integer = x(0).Length
            Dim numSamples As Integer = x.Length
            ' Calculate the mean and variance
            Dim mean As Double() = New Double(numInputs - 1) {}
            Dim variance As Double() = New Double(numInputs - 1) {}

            For i As Integer = 0 To numInputs - 1
                Dim sum As Double = 0
                For j As Integer = 0 To numSamples - 1
                    sum += x(j)(i)
                Next
                mean(i) = sum / numSamples

                Dim squaredSum As Double = 0
                For j As Integer = 0 To numSamples - 1
                    squaredSum += (x(j)(i) - mean(i)) * (x(j)(i) - mean(i))
                Next
                variance(i) = squaredSum / numSamples
            Next

            ' Normalize the inputs
            Dim normalizedX As Double()() = New Double(numSamples - 1)() {}
            For i As Integer = 0 To numSamples - 1
                normalizedX(i) = New Double(numInputs - 1) {}
                For j As Integer = 0 To numInputs - 1
                    normalizedX(i)(j) = (x(i)(j) - mean(j)) / Math.Sqrt(variance(j) + epsilon)
                Next
            Next

            Return normalizedX
        End Function
        Private Function AddResidual(x As Double()(), residual As Double()()) As Double()()
            Dim numInputs As Integer = x(0).Length
            Dim numSamples As Integer = x.Length
            Dim result As Double()() = New Double(numSamples - 1)() {}
            For i As Integer = 0 To numSamples - 1
                result(i) = New Double(numInputs - 1) {}
                For j As Integer = 0 To numInputs - 1
                    result(i)(j) = x(i)(j) + residual(i)(j)
                Next
            Next

            Return result
        End Function
        ' Positional encoding function
        Private Function GetPositionalEncoding(inputLength As Integer, dimModel As Integer) As Double()()
            Dim positionalEncoding As Double()() = New Double(inputLength - 1)() {}
            For i As Integer = 0 To inputLength - 1
                positionalEncoding(i) = New Double(dimModel - 1) {}
                For j As Integer = 0 To dimModel - 1
                    If j Mod 2 = 0 Then
                        positionalEncoding(i)(j) = Math.Sin(i / Math.Pow(10000, j / dimModel))

                    Else positionalEncoding(i)(j) = Math.Cos(i / Math.Pow(10000, (j - 1) / dimModel))
                    End If
                Next
            Next
            Return positionalEncoding
        End Function

        'Masking Function for the decoder self-attention 
        Private Function MaskSelfAttention(inputs As Double()(), maskValue As Double) As Double()()
            Dim numSamples As Integer = inputs.Length
            Dim numSteps As Integer = inputs(0).Length
            Dim maskedInputs As Double()() = New Double(numSamples - 1)() {}
            For i As Integer = 0 To numSamples - 1
                maskedInputs(i) = New Double(numSteps - 1) {}
                For j As Integer = 0 To numSteps - 1
                    If j < i Then
                        maskedInputs(i)(j) = maskValue
                    Else
                        maskedInputs(i)(j) = inputs(i)(j)
                    End If
                Next
            Next

            Return maskedInputs
        End Function

        Private Function ConvertToNumericData(ByRef inputText As List(Of List(Of String)), ByRef vocab As Dictionary(Of String, Integer)) As List(Of List(Of Integer))
            Dim numericData As New List(Of List(Of Integer))

            For Each sentence In inputText
                Dim numericSentence As New List(Of Integer)
                For Each word In sentence
                    If vocab.ContainsKey(word) Then
                        numericSentence.Add(vocab(word))
                    Else
                        numericSentence.Add(vocab("<UNK>")) ' Replace unknown words with the <UNK> token.
                    End If
                Next
                numericData.Add(numericSentence)
            Next

            Return numericData
        End Function
        Public Function EncodeQuery(input As List(Of List(Of Double))) As List(Of List(Of Double))
            ' Step 1: Perform Multi-Headed Attention
            Dim attentionOutput As List(Of List(Of Double)) = MultiHeadAttention.Attention(input)

            ' Step 2: Add Residual Connection and Normalize
            Dim attentionOutputWithResidual As List(Of List(Of Double)) = AddAndNormalize(input, attentionOutput)

            ' Step 3: Perform Feed Forward Network

            Dim feedForwardOutput As New List(Of List(Of Double))
            For Each item In attentionOutputWithResidual
                Dim iFeedForwardNetwork = New FeedForwardNetwork(EmbeddingSize, 8)

                feedForwardOutput.Add(iFeedForwardNetwork.Forward(item).ToList)
            Next


            ' Step 4: Add Residual Connection and Normalize
            Dim output As List(Of List(Of Double)) = AddAndNormalize(attentionOutputWithResidual, feedForwardOutput)

            Return output
        End Function
        Private Function AddAndNormalize(input As List(Of List(Of Double)), output As List(Of List(Of Double))) As List(Of List(Of Double))
            ' Add Residual Connection
            Dim residual As List(Of List(Of Double)) = MathMult.ConcatenateMatrices(input, output, MathMult.ConcatenationType.Vertical)

            ' Layer Normalization
            Dim normalized As List(Of List(Of Double)) = LayerNormalization.Normalize(residual)

            Return normalized
        End Function
        Public Function GetPredictions(ByRef InputQuery_Q As List(Of List(Of Double)), Key_K As List(Of List(Of Double)), TargetQuery_V As List(Of List(Of Double)))
            'stage 2 -(Merge Inputs and Targets - to create trainingPrediction)
            Dim Predictions As New List(Of List(Of Double))
            Dim BatchQuerys As New List(Of List(Of Double))

            'ADD INPUTS & TARGETS

            Dim Merge = MultiHeadAttention.MultiHeadedAttention(InputQuery_Q, Key_K, TargetQuery_V)
            Dim Add_B = MathMult.ConcatenateMatrices(TargetQuery_V, Merge, MathMult.ConcatenationType.Vertical)
            'ff
            For Each Query In Add_B
                Dim iFeedForwardNetwork = New FeedForwardNetwork(EmbeddingSize, 8)

                BatchQuerys.Add(iFeedForwardNetwork.Forward(Query).ToList)
            Next
            'add
            Predictions = MathMult.ConcatenateMatrices(BatchQuerys, Add_B, MathMult.ConcatenationType.Vertical)

        End Function

        Public Sub Train(ByRef BatchInput As List(Of List(Of Double)), ByRef BatchTargets As List(Of List(Of Double)))



            'Stage 1
            'Input encoder (Inputs)
            Dim InputQuery_Q = EncodeInputs(BatchInput)
            Dim Key_K = MathMult.TransposeMatrix(InputQuery_Q)
            'Enables for Prediction to focus on next term
            Dim TargetQuery_V = EncodeTargets(BatchTargets)

            Dim Predictions As List(Of List(Of Double)) = GetPredictions(InputQuery_Q, Key_K, TargetQuery_V)
            'Refine Predictions
            Dim Lin As New LinearLayer(Predictions(0).Count, BatchInput(0).Count)
            Predictions = Lin.Forward(Predictions)
            'SoftMax
            Predictions = MathMult.Softmax(Predictions)

            Dim loss = CrossEntropyLoss.ComputeCrossEntropyLoss(Predictions, TargetQuery_V)
            ' Compute the gradients of the loss with respect to the model's parameters.
            Dim gradients = CrossEntropyLoss.ComputeGradients(Predictions, BatchTargets)

            ' Calculate and display the average loss for the epoch.
            Dim averageLoss = loss / BatchInput.Count

        End Sub

        Private Sub UpdateParametersWithGradientDescent(parameters As List(Of List(Of Double)),
                                                gradients As List(Of List(Of Double)),
                                                learningRate As Double)
            ' Perform gradient descent update for each parameter in the model.
            For i = 0 To parameters.Count - 1
                For j = 0 To parameters(i).Count - 1
                    ' Update parameter using the negative gradient direction.
                    parameters(i)(j) -= learningRate * gradients(i)(j)
                Next
            Next
        End Sub

        Public Function Forward(ByRef BatchInput As List(Of List(Of Double))) As List(Of List(Of Double))

            'Stage 1
            'Input encoder (Inputs)
            Dim InputQuery_Q = EncodeQuery(BatchInput)
            Dim Predictions As List(Of List(Of Double)) = EncodeQuery(InputQuery_Q)

            'Refine Predictions
            'Linear
            Dim Lin As New LinearLayer(Predictions(0).Count, BatchInput(0).Count)
            Predictions = Lin.Forward(Predictions)
            'SoftMax
            Predictions = MathMult.Softmax(Predictions)
            Return Predictions
        End Function
        Public Function EncodeInputs(ByRef BatchInput As List(Of List(Of Double))) As List(Of List(Of Double))
#Region "Encoder Stage 1 - Iterate 2"
            Dim BatchQuerys As New List(Of List(Of Double))

            'Mutli-headed Attention
            Dim Weights = MultiHeadAttention.MultiHeadedAttention(BatchInput)
            'Add + Normalize
            Dim AddLayer1 = MathMult.ConcatenateMatrices(BatchInput, Weights, MathMult.ConcatenationType.Vertical)

            'Feed Forwards
            For Each Query In AddLayer1
                Dim iFeedForwardNetwork = New FeedForwardNetwork(EmbeddingSize, 8)

                BatchQuerys.Add(iFeedForwardNetwork.Forward(Query).ToList)

            Next
            'Add + Normalize
            Dim AddLayer2 = MathMult.ConcatenateMatrices(BatchQuerys, AddLayer1, MathMult.ConcatenationType.Vertical)
#End Region
            Return AddLayer2
        End Function

        Private Function EncodeTargets(ByRef BatchInput As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim BatchQuerys As New List(Of List(Of Double))

            'Mutli-headed Attention
            Dim Weights = MultiHeadAttention.MultiHeadedAttention(BatchInput)
            'Add + Normalize
            Dim AddLayer1 = MathMult.ConcatenateMatrices(BatchInput, Weights, MathMult.ConcatenationType.Vertical)

            Return AddLayer1
        End Function

    End Class
    Public Class LinearLayer
        Dim rand As New Random()

        Private weights As List(Of List(Of Double))
        Private bias As List(Of Double)

        Public Sub New(inputSize As Integer, outputSize As Integer)
            Randomize()
            weights = InitializeWeightMatrix(inputSize, outputSize)
            bias = New List(Of Double)
            For i As Integer = 0 To outputSize - 1


                bias.Add(rand.Next(-1, 1.0))
            Next
        End Sub
        Public Shared Function InitializeWeightMatrix(ByVal inputSize As Integer, ByVal outputSize As Integer) As List(Of List(Of Double))
            Dim weights As List(Of List(Of Double)) = New List(Of List(Of Double))
            Dim random As Random = New Random()

            For i As Integer = 0 To inputSize - 1
                Dim row As List(Of Double) = New List(Of Double)
                For j As Integer = 0 To outputSize - 1
                    row.Add(random.NextDouble())
                Next
                weights.Add(row)
            Next

            Return weights
        End Function

        Public Function Forward(input As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim output As New List(Of List(Of Double))()

            For Each inputData In input
                Dim weightedSum As New List(Of Double)()

                For i As Integer = 0 To weights.Count - 1
                    Dim weightRow As List(Of Double) = weights(i)
                    Dim weightedInput As Double = 0.0

                    For j As Integer = 0 To inputData.Count - 1
                        weightedInput += weightRow(j) * inputData(j)
                    Next

                    weightedSum.Add(weightedInput + bias(i))
                Next

                output.Add(weightedSum)
            Next

            Return output
        End Function


    End Class
    Public Class MathMult
        Public Enum ConcatenationType
            Horizontal
            Vertical
        End Enum
        Public Shared Function ConcatenateMatrices(matrix1 As List(Of List(Of Double)), matrix2 As List(Of List(Of Double)), concatenateVertical As ConcatenationType) As List(Of List(Of Double))
            Dim concatenatedMatrix As New List(Of List(Of Double))

            If concatenateVertical = ConcatenationType.Vertical Then
                ' Vertical concatenation
                concatenatedMatrix.AddRange(matrix1)
                concatenatedMatrix.AddRange(matrix2)
            Else
                ' Horizontal concatenation

                ' Ensure the matrices have the same number of rows
                If matrix1.Count <> matrix2.Count Then
                    Throw New ArgumentException("Matrices must have the same number of rows.")
                End If

                ' Concatenate the rows of matrix1 and matrix2 side by side
                For rowIndex As Integer = 0 To matrix1.Count - 1
                    Dim concatenatedRow As New List(Of Double)
                    concatenatedRow.AddRange(matrix1(rowIndex))
                    concatenatedRow.AddRange(matrix2(rowIndex))
                    concatenatedMatrix.Add(concatenatedRow)
                Next
            End If

            Return concatenatedMatrix
        End Function
        Public Shared Function DotProduct(vector1 As List(Of Double), vector2 As List(Of Double)) As Double
            Dim result As Double = 0.0

            For i = 0 To vector1.Count - 1
                result += vector1(i) * vector2(i)
            Next

            Return result
        End Function
        Public Shared Function DotProduct(matrix1 As List(Of List(Of Double)), matrix2 As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As New List(Of List(Of Double))

            For i = 0 To matrix1.Count - 1
                Dim row As New List(Of Double)

                For j = 0 To matrix2(0).Count - 1
                    Dim sum As Double = 0.0

                    For k = 0 To matrix1(0).Count - 1
                        sum += matrix1(i)(k) * matrix2(k)(j)
                    Next

                    row.Add(sum)
                Next

                result.Add(row)
            Next

            Return result
        End Function
        Public Shared Function TransposeMatrix(matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim rows As Integer = matrix.Count
            Dim cols As Integer = matrix(0).Count

            Dim result As New List(Of List(Of Double))
            For i = 0 To cols - 1
                Dim newRow As New List(Of Double)
                For j = 0 To rows - 1
                    newRow.Add(matrix(j)(i))
                Next
                result.Add(newRow)
            Next

            Return result
        End Function
        Public Shared Function TransposeVector(vector As List(Of Double)) As List(Of Double)
            Dim result As New List(Of Double)

            For i = 0 To vector.Count - 1
                result.Add(vector(i))
            Next

            Return result
        End Function
        Public Shared Function ScaleMatrix(matrix As List(Of List(Of Double)), scaleFactor As Integer) As List(Of List(Of Double))
            Dim result As New List(Of List(Of Double))

            For i = 0 To matrix.Count - 1
                Dim newRow As New List(Of Double)
                For j = 0 To matrix(i).Count - 1
                    newRow.Add(matrix(i)(j) / scaleFactor)
                Next
                result.Add(newRow)
            Next

            Return result
        End Function
        Public Shared Function Softmax(vector As List(Of Double)) As List(Of Double)
            Dim result As New List(Of Double)
            Dim maxValue As Double = vector.Max()
            Dim expSum As Double = 0.0

            For i = 0 To vector.Count - 1
                expSum += Math.Exp(vector(i) - maxValue)
            Next

            For i = 0 To vector.Count - 1
                result.Add(Math.Exp(vector(i) - maxValue) / expSum)
            Next

            Return result
        End Function
        Public Shared Function Softmax(matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim result As New List(Of List(Of Double))

            For i = 0 To matrix.Count - 1
                Dim row As New List(Of Double)
                Dim maxValue As Double = matrix(i).Max()
                Dim expSum As Double = 0.0

                For j = 0 To matrix(i).Count - 1
                    expSum += Math.Exp(matrix(i)(j) - maxValue)
                Next

                For j = 0 To matrix(i).Count - 1
                    row.Add(Math.Exp(matrix(i)(j) - maxValue) / expSum)
                Next

                result.Add(row)
            Next

            Return result
        End Function

        Public Class SGD
            Private ReadOnly parameters As List(Of List(Of Double))
            Private ReadOnly learningRate As Double

            Public Sub New(parameters As List(Of List(Of Double)), learningRate As Double)
                Me.parameters = parameters
                Me.learningRate = learningRate
            End Sub

            Public Sub StepForwards(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double)))
                Dim gradients As List(Of List(Of Double)) = ComputeGradients(predictions, targets)

                For i As Integer = 0 To parameters.Count - 1
                    Dim parameterRow As List(Of Double) = parameters(i)
                    Dim gradientRow As List(Of Double) = gradients(i)

                    For j As Integer = 0 To parameterRow.Count - 1
                        parameterRow(j) -= learningRate * gradientRow(j)
                    Next
                Next
            End Sub

            Private Shared Function CreateRandomList(rows As Integer, columns As Integer) As List(Of List(Of Double))
                Dim random As New Random()
                Dim matrix As New List(Of List(Of Double))()

                For i As Integer = 0 To rows - 1
                    Dim row As New List(Of Double)()

                    For j As Integer = 0 To columns - 1
                        row.Add(random.NextDouble())
                    Next

                    matrix.Add(row)
                Next

                Return matrix
            End Function

            Private Function ComputeGradients(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim gradients As New List(Of List(Of Double))()

                For i As Integer = 0 To predictions.Count - 1
                    Dim predictionRow As List(Of Double) = predictions(i)
                    Dim targetRow As List(Of Double) = targets(i)
                    Dim gradientRow As New List(Of Double)()

                    For j As Integer = 0 To predictionRow.Count - 1
                        gradientRow.Add(predictionRow(j) - targetRow(j))
                    Next

                    gradients.Add(gradientRow)
                Next

                Return gradients
            End Function
        End Class
    End Class
    Public Class MultiHeadAttention
        Private ReadOnly headCount As Integer
        Private ReadOnly headSize As Integer
        Private ReadOnly modelSize As Integer
        Private ReadOnly queryWeight()() As Double
        Private ReadOnly keyWeight()() As Double
        Private ReadOnly valueWeight()() As Double
        Private ReadOnly outputWeight()() As Double

        Public Sub New(headCount As Integer, modelSize As Integer)
            Me.headCount = headCount
            Me.modelSize = modelSize
            Me.headSize = modelSize \ headCount

            ' Initialize weights
            queryWeight = New Double(headCount - 1)() {}
            keyWeight = New Double(headCount - 1)() {}
            valueWeight = New Double(headCount - 1)() {}
            outputWeight = New Double(headCount - 1)() {}

            For i = 0 To headCount - 1
                queryWeight(i) = New Double(modelSize - 1) {}
                keyWeight(i) = New Double(modelSize - 1) {}
                valueWeight(i) = New Double(modelSize - 1) {}
                outputWeight(i) = New Double(modelSize - 1) {}

                ' Initialize weights with random values or desired initialization logic
                For j = 0 To modelSize - 1
                    queryWeight(i)(j) = GetRandomValue()
                    keyWeight(i)(j) = GetRandomValue()
                    valueWeight(i)(j) = GetRandomValue()
                    outputWeight(i)(j) = GetRandomValue()
                Next
            Next
        End Sub

        Public Function Forward(input As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim batchSize = input.Count
            '  Dim output = New Double(batchSize - 1)() {}
            Dim output As New List(Of List(Of Double))
            For i = 0 To batchSize - 1
                Dim inputSequence = input(i)
                Dim sequenceLength = inputSequence.Count

                ' Split input sequence into heads
                Dim inputHeads = SplitSequenceIntoHeads(inputSequence.ToArray)

                ' Apply attention to each head
                'Dim headOutputs = New Double(headCount - 1)() {}
                Dim headOutputs = New List(Of List(Of Double))

                For j = 0 To headCount - 1
                    Dim query = ApplyLinearTransformation(inputHeads(j), queryWeight(j))
                    Dim key = ApplyLinearTransformation(inputHeads(j), keyWeight(j))
                    Dim value = ApplyLinearTransformation(inputHeads(j), valueWeight(j))

                    ' Compute attention scores and attention weights
                    Dim attentionScores = ComputeAttentionScores(query, key)
                    Dim attentionWeights = Softmax(attentionScores)

                    ' Compute the weighted sum of values using attention weights
                    Dim iweightedSum = WeightedSum(value, attentionWeights)

                    ' Apply linear transformation and store the output for the current head
                    headOutputs(j) = ApplyLinearTransformation(iweightedSum, outputWeight(j)).ToList
                Next

                ' Concatenate head outputs and apply final linear transformation
                output(i) = ConcatenateHeadsAndApplyLinearTransformation(headOutputs).ToList
            Next

            Return output
        End Function

        'Single Forwards function is Required (list (of double)) for Single Pass



        Private Function SplitSequenceIntoHeads(sequence As Double()) As Double()()
            Dim headCount = queryWeight.Length
            Dim headSize = sequence.Length \ headCount
            Dim heads = New Double(headCount - 1)() {}

            For i = 0 To headCount - 1
                heads(i) = New Double(headSize - 1) {}
                Array.Copy(sequence, i * headSize, heads(i), 0, headSize)
            Next

            Return heads
        End Function

        Public Shared Function ApplyLinearTransformation(sequence As Double(), weights As Double()) As Double()
            Dim output = New Double(sequence.Length - 1) {}

            For i = 0 To sequence.Length - 1
                Dim sum = 0.0
                For j = 0 To weights.Length - 1
                    sum += sequence(j) * weights(j)
                Next
                output(i) = sum
            Next

            Return output
        End Function

        Private Function ComputeAttentionScores(query As Double(), key As Double()) As Double()
            Dim scores = New Double(query.Length - 1) {}

            For i = 0 To query.Length - 1
                scores(i) = query(i) * key(i)
            Next

            Return scores
        End Function

        Public Function Softmax(scores As Double()) As Double()
            Dim softmaxScores = New Double(scores.Length - 1) {}

            Dim maxScore = Double.MinValue
            For i = 0 To scores.Length - 1
                If scores(i) > maxScore Then
                    maxScore = scores(i)
                End If
            Next

            Dim expSum = 0.0
            For i = 0 To scores.Length - 1
                expSum += Math.Exp(scores(i) - maxScore)
            Next

            For i = 0 To scores.Length - 1
                softmaxScores(i) = Math.Exp(scores(i) - maxScore) / expSum
            Next

            Return softmaxScores
        End Function

        Private Function WeightedSum(values As Double(), weights As Double()) As Double()
            Dim sum() As Double = {0.0}
            For i = 0 To values.Length - 1
                sum(i) += values(i) * weights(i)
            Next

            Return sum
        End Function

        Private Function ConcatenateHeadsAndApplyLinearTransformation(headOutputs As List(Of List(Of Double))) As Double()
            Dim outputSize = outputWeight(0).Length
            Dim output = New Double(outputSize - 1) {}

            Dim index = 0
            For i = 0 To headOutputs.Count - 1
                For j = 0 To headOutputs(i).Count - 1
                    output(index) = headOutputs(i)(j)
                    index += 1
                Next
            Next

            output = ApplyLinearTransformation(output, outputWeight(0))
            Return output
        End Function

        Private Function GetRandomValue() As Double
            ' Use your desired logic for weight initialization
            Dim rnd = New Random()
            Return rnd.NextDouble() - 0.5
        End Function

        ''' <summary>
        ''' Input to this layer will be repeated for each head ,
        '''Query , Key and value,
        '''On entry the Input should be
        '''the key should be transposed first then passed through the linear layer
        '''the weights should be shared between inputs.
        '''The output for each q/k/v matrix ... should be as follows
        '''Value Remains untouched ,
        '''The A Output is created by the dot-product
        '''                         (of the output for the key and query from the linear layer)
        '''This output is scaled by the dimension of the key_vector
        '''And finally passed through a soft-max . This output is to be ,
        '''DotProduct with the Value matrix outputted from the linear layer
        '''To produce the final output for the Attention Head.
        ''' </summary>
        ''' <param name="input"></param>
        ''' <returns></returns>
        Public Shared Function Attention(ByRef input As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim batchSize As Integer = input.Count
            Dim inputSize As Integer = input(0).Count
            '1.Create Q , K, V
            '-> Q)Pass Q through linear layer
            '-> K)Transpose K and Pass through liner layer
            '-> V)Copy Q to V
            Dim Linear As New LinearLayer(inputSize, batchSize)
            Dim query As List(Of List(Of Double)) = Linear.Forward(input)
            Dim key As List(Of List(Of Double)) = MathMult.TransposeMatrix(input)
            key = Linear.Forward(key)
            Dim value As List(Of List(Of Double)) = query
            '  Create dotProduct of Key*Query
            ' -> Scale by K(Dimension)
            ' -> SoftMax = AttentionOutput
            Dim attentionOutput As List(Of List(Of Double)) = MathMult.Softmax(MathMult.ScaleMatrix(MathMult.DotProduct(query, key), key.Count))
            'Create DotProduct of (V)Value & AttentionOutput
            'Return Attention head


            Return MathMult.DotProduct(attentionOutput, value)
        End Function
        Public Shared Function Mask(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim rows As Integer = matrix.Count
            Dim cols As Integer = matrix(0).Count

            Dim result As New List(Of List(Of Double))

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i)(j) = matrix(i)(j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Function MultiHeadedAttention(ByRef Input As List(Of List(Of Double)), Optional HeadCount As Integer = 8) As List(Of List(Of Double))
            Dim headSize As Integer = Input(0).Count \ HeadCount
            Dim heads As New List(Of List(Of List(Of Double)))
            For i = 0 To HeadCount - 1
                heads.Add(Attention(Input))
            Next
            Dim Output As New List(Of List(Of Double))
            For Each item In heads
                Output = MathMult.ConcatenateMatrices(Output, item, MathMult.ConcatenationType.Vertical)
            Next
            ' Apply linear transformation to obtain final output
            Dim finalOutput As List(Of List(Of Double)) = New LinearLayer(HeadCount * headSize, Input(0).Count).Forward(Output)

            Return finalOutput

        End Function


        Public Shared Function MultiHeadedAttention(ByRef Q As List(Of List(Of Double)), ByRef K As List(Of List(Of Double)), ByRef V As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim headSize As Integer = V(0).Count \ 3
            Dim heads As New List(Of List(Of List(Of Double)))
            heads.Add(Attention(Q))
            heads.Add(Attention(K))
            heads.Add(Attention(V))

            Dim Output As New List(Of List(Of Double))
            For Each item In heads
                Output = MathMult.ConcatenateMatrices(Output, item, MathMult.ConcatenationType.Vertical)
            Next
            ' Apply linear transformation to obtain final output
            Dim finalOutput As List(Of List(Of Double)) = New LinearLayer(3 * headSize, V(0).Count).Forward(Output)

            Return finalOutput

        End Function
    End Class
    Public Class Tril
        Public Sub Main()
            Dim matrix(,) As Integer = {{1, 2, 3, 9}, {4, 5, 6, 8}, {7, 8, 9, 9}}

            Dim result(,) As Integer = Tril(matrix)

            Console.WriteLine("Matrix:")
            PrintMatrix(matrix)

            Console.WriteLine("Tril Result:")
            PrintMatrix(result)
            Console.ReadLine()
        End Sub


        Public Shared Function Tril(ByVal matrix(,) As Integer) As Integer(,)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            Dim result(rows - 1, cols - 1) As Integer

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i, j) = matrix(i, j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Function Tril(ByVal matrix(,) As Double) As Double(,)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            Dim result(rows - 1, cols - 1) As Double

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i, j) = matrix(i, j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Function Tril(ByVal matrix As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim rows As Integer = matrix.Count
            Dim cols As Integer = matrix(0).Count

            Dim result As New List(Of List(Of Double))

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    If j <= i Then
                        result(i)(j) = matrix(i)(j)
                    End If
                Next
            Next

            Return result
        End Function
        Public Shared Sub PrintMatrix(ByVal matrix(,) As Double)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    Console.Write(matrix(i, j) & " ")
                Next
                Console.WriteLine()
            Next
        End Sub
        Public Shared Sub PrintMatrix(ByVal matrix(,) As Integer)
            Dim rows As Integer = matrix.GetLength(0)
            Dim cols As Integer = matrix.GetLength(1)

            For i As Integer = 0 To rows - 1
                For j As Integer = 0 To cols - 1
                    Console.Write(matrix(i, j) & " ")
                Next
                Console.WriteLine()
            Next
        End Sub
    End Class
    Public Class Softmax
        Public Shared Function Softmax(matrix2 As Integer(,)) As Double(,)
            Dim numRows As Integer = matrix2.GetLength(0)
            Dim numColumns As Integer = matrix2.GetLength(1)

            Dim softmaxValues(numRows - 1, numColumns - 1) As Double

            ' Compute softmax values for each row
            For i As Integer = 0 To numRows - 1
                Dim rowSum As Double = 0

                ' Compute exponential values and sum of row elements
                For j As Integer = 0 To numColumns - 1
                    softmaxValues(i, j) = Math.Sqrt(Math.Exp(matrix2(i, j)))
                    rowSum += softmaxValues(i, j)
                Next

                ' Normalize softmax values for the row
                For j As Integer = 0 To numColumns - 1
                    softmaxValues(i, j) /= rowSum
                Next
            Next

            ' Display the softmax values
            Console.WriteLine("Calculated:" & vbNewLine)
            For i As Integer = 0 To numRows - 1
                For j As Integer = 0 To numColumns - 1

                    Console.Write(softmaxValues(i, j).ToString("0.0000") & " ")
                Next
                Console.WriteLine(vbNewLine & "---------------------")
            Next
            Return softmaxValues
        End Function
        Public Shared Sub Main()
            Dim input() As Double = {1.0, 2.0, 3.0}

            Dim output() As Double = Softmax(input)

            Console.WriteLine("Input: {0}", String.Join(", ", input))
            Console.WriteLine("Softmax Output: {0}", String.Join(", ", output))
            Console.ReadLine()
        End Sub

        Public Shared Function Softmax(ByVal input() As Double) As Double()
            Dim maxVal As Double = input.Max()

            Dim exponentiated() As Double = input.Select(Function(x) Math.Exp(x - maxVal)).ToArray()

            Dim sum As Double = exponentiated.Sum()

            Dim softmaxOutput() As Double = exponentiated.Select(Function(x) x / sum).ToArray()

            Return softmaxOutput
        End Function
    End Class
    Public Class ScaledDotProductAttention
        Private ReadOnly headSize As Integer
        Private ReadOnly numHeads As Integer

        Public Sub New(headSize As Integer, numHeads As Integer)
            Me.headSize = headSize
            Me.numHeads = numHeads
        End Sub

        Public Function Forward(queries As List(Of List(Of Double)), keys As List(Of List(Of Double)), values As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim batchSize = queries.Count
            Dim querySize = queries(0).Count
            Dim keySize = keys(0).Count
            Dim valueSize = values(0).Count

            Dim outputs As New List(Of List(Of Double))

            For i = 0 To batchSize - 1
                Dim query = queries(i)
                Dim key = keys(i)
                Dim value = values(i)

                Dim output = New List(Of Double)

                For j = 0 To valueSize - 1
                    Dim weightedSum As Double

                    For k = 0 To numHeads - 1
                        Dim qStart = k * headSize
                        Dim qEnd = qStart + headSize - 1
                        Dim kStart = k * headSize
                        Dim kEnd = kStart + headSize - 1
                        Dim vStart = k * headSize
                        Dim vEnd = vStart + headSize - 1

                        For l = 0 To headSize - 1
                            weightedSum += (query(qStart + l) * key(kStart + l) * value(vStart + l))
                        Next
                    Next

                    output(j) = weightedSum
                Next

                outputs(i) = output
            Next

            Return outputs
        End Function
    End Class
    Public Class LayerNormalization
        Private epsilon As Double
        Private iscale() As Double
        Private ibias() As Double
        Private ReadOnly hiddenSize As Integer


        Public Sub New(featureSize As Integer, epsilon As Double)
            Me.epsilon = epsilon

            iscale = New Double(featureSize - 1) {}
            ibias = New Double(featureSize - 1) {}

            ' Initialize scale and bias with desired values or random initialization logic
            For i = 0 To featureSize - 1
                iscale(i) = 1.0
                ibias(i) = 0.0
            Next
        End Sub

        Public Function Forward(inputs As Double()()) As Double()()
            Dim output = New Double(inputs.Length - 1)() {}

            For i = 0 To inputs.Length - 1
                Dim inputSequence = inputs(i)
                Dim sequenceLength = inputSequence.Length

                Dim mean = ComputeMean(inputSequence)
                Dim variance = ComputeVariance(inputSequence, mean)
                Dim normalized = Normalize(inputSequence, mean, variance)

                Dim scaled = Scale(normalized)
                Dim biased = Bias(scaled)

                output(i) = biased
            Next

            Return output
        End Function

        Private Function ComputeMean(sequence As Double()) As Double
            Dim sum = 0.0
            For Each value In sequence
                sum += value
            Next

            Return sum / sequence.Length
        End Function

        Private Shared Function ComputeVariance(sequence As Double(), mean As Double) As Double
            Dim sum = 0.0
            For Each value In sequence
                sum += (value - mean) * (value - mean)
            Next

            Return sum / sequence.Length
        End Function

        Public Shared Function Normalize(sequence As Double(), mean As Double, variance As Double) As Double()
            Dim normalized = New Double(sequence.Length - 1) {}

            Dim stdDev = Math.Sqrt(variance)
            For i = 0 To sequence.Length - 1
                normalized(i) = (sequence(i) - mean) / stdDev
            Next

            Return normalized
        End Function
        Public Shared Function Normalize(inputs As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim normalizedOutputs As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

            For Each inputVector As List(Of Double) In inputs
                Dim mean As Double = inputVector.Average()
                Dim variance = ComputeVariance(inputVector.ToArray, mean)
                Dim stdDev As Double = Math.Sqrt(variance)

                Dim normalizedVector As List(Of Double) = inputVector.Select(Function(x) (x - mean) / stdDev).ToList()
                normalizedOutputs.Add(normalizedVector)
            Next

            Return normalizedOutputs
        End Function

        Private Function Scale(sequence As Double()) As Double()
            Dim scaled = New Double(sequence.Length - 1) {}

            For i = 0 To sequence.Length - 1
                scaled(i) = sequence(i) * iscale(i)
            Next

            Return scaled
        End Function

        Private Function Bias(sequence As Double()) As Double()
            Dim biased = New Double(sequence.Length - 1) {}

            For i = 0 To sequence.Length - 1
                biased(i) = sequence(i) + ibias(i)
            Next

            Return biased
        End Function
    End Class
    Public Class FeedForwardNetwork
        Private ReadOnly inputSize As Integer
        Private ReadOnly hiddenSize As Integer
        Private ReadOnly outputSize As Integer
        Private ReadOnly hiddenWeight()() As Double
        Private ReadOnly hiddenBias() As Double
        Private ReadOnly outputWeight()() As Double
        Private ReadOnly outputBias() As Double

        Public Sub New(inputSize As Integer, hiddenSize As Integer)
            Me.inputSize = inputSize
            Me.hiddenSize = hiddenSize
            Me.outputSize = inputSize

            hiddenWeight = New Double(inputSize - 1)() {}
            hiddenBias = New Double(inputSize - 1) {}
            outputWeight = New Double(hiddenSize - 1)() {}
            outputBias = New Double(hiddenSize - 1) {}

            For i = 0 To inputSize - 1
                hiddenWeight(i) = New Double(hiddenSize - 1) {}
                For j = 0 To hiddenSize - 1
                    hiddenWeight(i)(j) = GetRandomValue()
                Next
                hiddenBias(i) = GetRandomValue()
            Next

            For i = 0 To hiddenSize - 1
                outputWeight(i) = New Double(outputSize - 1) {}
                For j = 0 To outputSize - 1
                    outputWeight(i)(j) = GetRandomValue()
                Next
                outputBias(i) = GetRandomValue()
            Next
        End Sub

        Public Function Forward(inputs As List(Of Double)) As Double()
            Dim hiddenOutputs = ApplyLinearTransformation(inputs.ToArray, hiddenWeight, hiddenBias)
            Dim hiddenActivated = ReLU(hiddenOutputs)
            Dim outputs = ApplyLinearTransformation(hiddenActivated, outputWeight, outputBias)

            Return outputs
        End Function

        '' batch forwards function Required for Transformer Training function


        Private Shared Function ApplyLinearTransformation(inputs As Double(), weights As Double()(), bias As Double()) As Double()
            Dim outputs = New Double(weights.Length - 1) {}

            For i = 0 To weights.Length - 1
                Dim sum = 0.0
                For j = 0 To inputs.Length - 1
                    sum += inputs(j) * weights(j)(i)
                Next
                outputs(i) = sum + bias(i)
            Next

            Return outputs
        End Function

        Private Shared Function ReLU(inputs As Double()) As Double()
            Dim outputs = New Double(inputs.Length - 1) {}

            For i = 0 To inputs.Length - 1
                outputs(i) = Math.Max(0.0, inputs(i))
            Next

            Return outputs
        End Function

        Private Function GetRandomValue() As Double
            Dim rnd = New Random()
            Return rnd.NextDouble() - 0.5
        End Function
    End Class
    Public Class CrossEntropyLoss
        Public Shared Function ComputeCrossEntropyLoss(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double))) As Double
            ' Ensure predictions and targets have the same shape
            If predictions.Count <> targets.Count OrElse predictions(0).Count <> targets(0).Count Then
                Throw New ArgumentException("Predictions and targets must have the same shape.")
            End If

            ' Compute the element-wise negative log likelihood
            Dim elementWiseLoss As New List(Of List(Of Double))()
            For i As Integer = 0 To predictions.Count - 1
                Dim lossRow As New List(Of Double)()
                For j As Integer = 0 To predictions(i).Count - 1
                    Dim p As Double = predictions(i)(j)
                    Dim t As Double = targets(i)(j)
                    Dim lossValue As Double = If(p > 0, -Math.Log(p) * t, 0)
                    lossRow.Add(lossValue)
                Next
                elementWiseLoss.Add(lossRow)
            Next

            ' Sum the losses across all elements and take the mean
            Dim totalLoss As Double = 0
            Dim numElements As Integer = 0
            For Each row In elementWiseLoss
                For Each lossValue In row
                    totalLoss += lossValue
                    numElements += 1
                Next
            Next
            Dim averageLoss As Double = totalLoss / numElements

            Return averageLoss
        End Function
        Public Shared Function ComputeGradients(predictions As List(Of List(Of Double)), targets As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim gradients As New List(Of List(Of Double))

            ' Iterate over each prediction and target pair
            For i As Integer = 0 To predictions.Count - 1
                Dim prediction As List(Of Double) = predictions(i)
                Dim target As List(Of Double) = targets(i)

                Dim gradient As New List(Of Double)

                ' Calculate gradient for each element in the prediction
                For j As Integer = 0 To prediction.Count - 1
                    ' Compute the gradient of the cross-entropy loss with respect to each prediction element
                    Dim grad As Double = prediction(j) - target(j)
                    gradient.Add(grad)
                Next

                gradients.Add(gradient)
            Next

            Return gradients
        End Function
        Public Function Forward(predictions As List(Of Double), targets As List(Of Double)) As Double
            Dim loss As Double = 0.0

            For i = 0 To predictions.Count - 1
                loss += targets(i) * Log(predictions(i))
            Next

            Return -loss
        End Function
    End Class

End Namespace