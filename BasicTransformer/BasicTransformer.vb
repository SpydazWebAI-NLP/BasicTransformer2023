Namespace iTransform
    Public Class BasicTransformer
        ' Transformer model configuration

        Private WordEmbeddings As Embeddings.Factory.WordEmbeddingsModel
        Private EmbeddingSize As Integer


        Public Sub New(embeddingSize As Integer, vocabulary As List(Of String))
            Me.EmbeddingSize = embeddingSize


            ' Initialize the word embeddings model
            WordEmbeddings = New Embeddings.Factory.HybridWordEmbeddingsModel(vocabulary)


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

        Public Function Train(ByRef BatchInput As List(Of List(Of Double)), ByRef BatchTargets As List(Of List(Of Double)))



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
            Return Predictions
            Dim loss = CrossEntropyLoss.ComputeCrossEntropyLoss(Predictions, TargetQuery_V)


            Return Predictions
        End Function

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

End Namespace