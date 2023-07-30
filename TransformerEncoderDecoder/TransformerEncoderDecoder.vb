Namespace TransformerEncoderDecoder
    Public Structure WordVector

        Public Word As String

        Private Shared irand As Random
        ' Encoder/Decoder)
        Private Dmodel As Integer
        'The length of this embedding is the dimensions of the model
        Private InputEmbeddingVector As List(Of Double)

        Private iPos As List(Of Double)

        Private iShot As List(Of Double)

        Private PreservedInput As List(Of Double)

        'ID of Word in Vocab
        Private VocabID As Integer
        ''' <summary>
        ''' The vocab Id produced by the tokenization
        ''' The modelSize(embedding Model(usually linked to Vocabulary length))
        ''' </summary>
        ''' <param name="ID">issued by the Tokenization input process </param>
        ''' <param name="Dmodel">512 is suggested to be the highest value</param>
        Public Sub New(ID As Integer, Optional Dmodel As Integer = 512)
            VocabID = ID
            Randomize()
            irand = New Random
            InputEmbeddingVector = New List(Of Double)
            Dmodel = Dmodel
            'Initalize our embeddings based on the model
            For i = 1 To Dmodel
                InputEmbeddingVector.Add(GenerateRandomWeight() + VocabID)
            Next

        End Sub

        Public Property InputEmbedding As List(Of Double)
            Get
                Return InputEmbeddingVector
            End Get
            Set(value As List(Of Double))
                InputEmbeddingVector = value
            End Set
        End Property

        ''' <summary>
        ''' Our Original Input embedding will be preserved for
        ''' later refocusing layers or training scenarios
        ''' </summary>
        ''' <returns></returns>
        Public ReadOnly Property ResdidualInput As List(Of Double)
            Get
                Return PreservedInput
            End Get
        End Property

        ''' <summary>
        ''' Provides one-shot sequence encoding for sentence
        ''' </summary>
        ''' <returns></returns>
        Private Property OneShotSequenceEncoding As List(Of Double)
            Set(value As List(Of Double))
                Dim OutputV As New List(Of Double)
                For i = 0 To Dmodel - 1
                    OutputV(i) = InputEmbeddingVector(i) + value(i)
                Next

                InputEmbeddingVector = OutputV
                iShot = value
            End Set
            Get
                Return iShot
            End Get
        End Property

        ''' <summary>
        ''' Must be applied at Sentence Level once added it will automatically
        ''' adjust the input embeddings accordingly
        ''' </summary>
        Private Property PositionalEncoding As List(Of Double)

            Set(value As List(Of Double))
                Dim OutputV As New List(Of Double)
                For i = 0 To Dmodel - 1
                    OutputV(i) = InputEmbeddingVector(i) + value(i)
                Next
                iPos = value
                InputEmbeddingVector = OutputV
            End Set
            Get
                Return iPos
            End Get
        End Property

        Public Shared Function GenerateRandomWeight() As Double
            Return irand.Next(0.0, 1.0)
        End Function

        Public Function CreateEmptyMatrix() As List(Of Double)
            'Create Empty matrix
            Dim encoding As New List(Of Double)
            For Each item In InputEmbeddingVector
                encoding.Add(0.0)
            Next
            Return encoding
        End Function

        ''' <summary>
        ''' to generate our correct embeddings
        ''' we need to calculate the Positional Encodings,and sentence sequence encodings 
        ''' </summary>
        ''' <param name="PositionalEncoding"></param>
        ''' <param name="WordIndex"></param>
        Public Sub CreateInputEmbedding(ByRef PositionalEncoding As List(Of Double),
                                                ByRef WordIndex As Integer)
            PreservedInput = New List(Of Double)

            PositionalEncoding = PositionalEncoding
            CreateOneShotSequenceEncoding(WordIndex)
            PreservedInput = InputEmbeddingVector
        End Sub

        Public Sub CreateOneShotSequenceEncoding(ByRef WordIndex As Integer)
            Dim encoding As List(Of Double) = CreateEmptyMatrix()


            ' Set the elements from index 0 to wordIndex-1 to 1
            For i As Integer = 0 To WordIndex - 1
                encoding(i) = 1
            Next

            OneShotSequenceEncoding = encoding
        End Sub

        ' Function to scale the InputEmbeddingVector by a scaleFactor
        Public Sub ScaleInputEmbedding(scaleFactor As Integer)
            For i = 0 To Dmodel - 1
                InputEmbeddingVector(i) /= scaleFactor
            Next
        End Sub
    End Structure
    Public Class PositionalEncoderDecoder
        Private ReadOnly encodingMatrix As List(Of List(Of Double))
        Private Vocabulary As New List(Of String)

        Public Sub New(maxLength As Integer, Dmodel As Integer, ByRef vocab As List(Of String))
            encodingMatrix = New List(Of List(Of Double))
            Vocabulary = vocab
            ' Create the encoding matrix
            For pos As Integer = 0 To maxLength - 1
                Dim encodingRow As List(Of Double) = New List(Of Double)()

                For i As Integer = 0 To Dmodel - 1
                    Dim angle As Double = pos / Math.Pow(10000, (2 * i) / Dmodel)
                    encodingRow.Add(Math.Sin(angle))
                    encodingRow.Add(Math.Cos(angle))
                Next

                encodingMatrix.Add(encodingRow)
            Next

            Vocabulary = vocab

        End Sub

        Public Function Decode(encodedInputs As List(Of List(Of Double))) As List(Of String)
            Dim decodedTokens As List(Of String) = New List(Of String)

            For Each encoding As List(Of Double) In encodedInputs
                ' Retrieve the token index based on the encoding
                Dim tokenIndex As Integer = GetTokenIndex(encoding)

                ' Retrieve the token based on the index
                If tokenIndex >= 0 Then
                    Dim token As String = GetToken(tokenIndex)
                    decodedTokens.Add(token)
                Else
                    ' Handle unknown encodings if necessary
                End If
            Next

            Return decodedTokens
        End Function

        Public Function Encode(inputTokens As List(Of String)) As List(Of List(Of Double))
            Dim encodedInputs As List(Of List(Of Double)) = New List(Of List(Of Double))()

            For pos As Integer = 0 To inputTokens.Count - 1
                Dim token As String = inputTokens(pos)
                Dim tokenEncoding As List(Of Double) = New List(Of Double)()

                ' Retrieve the positional encoding for the token
                tokenEncoding = encodingMatrix(pos)

                encodedInputs.Add(tokenEncoding)
            Next

            Return encodedInputs
        End Function

        Private Function GetToken(tokenIndex As Integer) As String
            ' Retrieve the token based on the index
            ' For simplicity, let's assume a fixed vocabulary
            Dim vocabulary As List(Of String) = GetVocabulary()

            If tokenIndex >= 0 AndAlso tokenIndex < vocabulary.Count Then
                Return vocabulary(tokenIndex)
            Else
                Return "Unknown" ' Unknown token
            End If
        End Function

        Private Function GetTokenIndex(token As String) As Integer
            ' Retrieve the index of the token in the vocabulary
            ' For simplicity, let's assume a fixed vocabulary
            Dim vocabulary As List(Of String) = GetVocabulary()
            Return vocabulary.IndexOf(token)
        End Function

        Private Function GetTokenIndex(encoding As List(Of Double)) As Integer
            ' Retrieve the index of the token based on the encoding
            ' For simplicity, let's assume a fixed vocabulary
            Dim vocabulary As List(Of String) = GetVocabulary()

            For i As Integer = 0 To encodingMatrix.Count - 1
                If encoding.SequenceEqual(encodingMatrix(i)) Then
                    Return i
                End If
            Next

            Return -1 ' Token not found
        End Function

        Private Function GetVocabulary() As List(Of String)
            ' Return the vocabulary list
            ' Modify this function as per your specific vocabulary
            Return Vocabulary
        End Function
    End Class
    Public Class TrainingDataGenerator

        ''' <summary>
        ''' The maximum sequence length for padding or truncating sequences.
        ''' </summary>
        Public Shared maxSequenceLength As Integer = 8

        ''' <summary>
        ''' The vocabulary used for encoding and decoding.
        ''' </summary>
        Public Shared TestVocab As New List(Of String)

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified maximum sequence length.
        ''' </summary>
        ''' <param name="MaxSeqLength">The maximum sequence length for padding or truncating sequences.</param>
        Public Sub New(ByRef MaxSeqLength As Integer)
            TestVocab = New List(Of String)
            maxSequenceLength = MaxSeqLength
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class.
        ''' </summary>
        Public Sub New()
            TestVocab = New List(Of String)
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified vocabulary.
        ''' </summary>
        ''' <param name="Vocab">The vocabulary used for encoding and decoding.</param>
        Public Sub New(ByRef Vocab As List(Of String))
            TestVocab = Vocab
        End Sub

        ''' <summary>
        ''' Initializes a new instance of the DataExtraction class with the specified vocabulary and maximum sequence length.
        ''' </summary>
        ''' <param name="Vocab">The vocabulary used for encoding and decoding.</param>
        ''' <param name="MaxSeqLength">The maximum sequence length for padding or truncating sequences.</param>
        Public Sub New(ByRef Vocab As List(Of String), ByRef MaxSeqLength As Integer)
            TestVocab = Vocab
            maxSequenceLength = MaxSeqLength
        End Sub

        ''' <summary>
        ''' Creates batches of data for training.
        ''' </summary>
        ''' <param name="trainingData">The training data as a list of string sequences.</param>
        ''' <param name="batchSize">The size of each batch.</param>
        Public Sub CreateData(ByRef trainingData As List(Of List(Of String)), ByRef batchSize As Integer)
            For batchStart As Integer = 0 To trainingData.Count - 1 Step batchSize
                Dim batchEnd As Integer = Math.Min(batchStart + batchSize - 1, trainingData.Count - 1)
                Dim batchInputs As List(Of List(Of Integer)) = GetBatchInputs(trainingData, batchStart, batchEnd)
                Dim batchTargets As List(Of List(Of Integer)) = GetBatchTargets(trainingData, batchStart, batchEnd)

                ' Perform further operations on the batches
            Next

            ' Compute loss
            ' Dim loss As Double = ComputeLoss(predictions, batchTargets)
        End Sub

        ''' <summary>
        ''' Converts a batch of data from a list of string sequences to a list of integer sequences.
        ''' </summary>
        ''' <param name="data">The input data as a list of string sequences.</param>
        ''' <param name="startIndex">The starting index of the batch.</param>
        ''' <param name="endIndex">The ending index of the batch.</param>
        ''' <returns>A list of integer sequences representing the batch inputs.</returns>
        Public Function GetBatchInputs(data As List(Of List(Of String)),
                                   startIndex As Integer,
                                   endIndex As Integer) As List(Of List(Of Integer))
            Dim batchInputs As New List(Of List(Of Integer))

            For i As Integer = startIndex To endIndex
                Dim sequence As List(Of String) = data(i)

                ' Convert words to corresponding indices
                Dim indices As List(Of Integer) = ConvertWordsToIndices(sequence)

                ' Pad or truncate sequence to the maximum length
                indices = PadOrTruncateSequence(indices, maxSequenceLength)

                ' Add the sequence to the batch
                batchInputs.Add(indices)
            Next

            Return batchInputs
        End Function

        ''' <summary>
        ''' Converts a batch of data from a list of string sequences to a list of integer sequences as targets.
        ''' </summary>
        ''' <param name="data">The input data as a list of string sequences.</param>
        ''' <param name="startIndex">The starting index of the batch.</param>
        ''' <param name="endIndex">The ending index of the batch.</param>
        ''' <returns>A list of integer sequences representing the batch targets.</returns>
        Public Function GetBatchTargets(data As List(Of List(Of String)), startIndex As Integer, endIndex As Integer) As List(Of List(Of Integer))
            Dim batchTargets As New List(Of List(Of Integer))

            For i As Integer = startIndex To endIndex
                Dim sequence As List(Of String) = data(i)

                ' Convert words to corresponding indices
                Dim indices As List(Of Integer) = ConvertWordsToIndices(sequence)

                ' Shift the sequence to get the target sequence
                Dim targetIndices As List(Of Integer) = ShiftSequence(indices)

                ' Pad or truncate sequence to the maximum length
                targetIndices = PadOrTruncateSequence(targetIndices, maxSequenceLength)

                ' Add the target sequence to the batch
                batchTargets.Add(targetIndices)
            Next

            Return batchTargets
        End Function

        ''' <summary>
        ''' Pads or truncates a sequence to a specified length.
        ''' </summary>
        ''' <param name="sequence">The input sequence.</param>
        ''' <param name="length">The desired length.</param>
        ''' <returns>The padded or truncated sequence.</returns>
        Public Function PadOrTruncateSequence(sequence As List(Of Integer), length As Integer) As List(Of Integer)
            If sequence.Count < length Then
                ' Pad the sequence with a special padding token
                sequence.AddRange(Enumerable.Repeat(TestVocab.IndexOf("PAD"), length - sequence.Count))
            ElseIf sequence.Count > length Then
                ' Truncate the sequence to the desired length
                sequence = sequence.GetRange(0, length)
            End If

            Return sequence
        End Function

        ''' <summary>
        ''' Shifts a sequence to the right and adds a special token at the beginning.
        ''' </summary>
        ''' <param name="sequence">The input sequence.</param>
        ''' <returns>The shifted sequence.</returns>
        Public Function ShiftSequence(sequence As List(Of Integer)) As List(Of Integer)
            ' Shifts the sequence to the right and adds a special token at the beginning
            Dim shiftedSequence As New List(Of Integer) From {TestVocab.IndexOf("START")}

            For i As Integer = 0 To sequence.Count - 1
                shiftedSequence.Add(sequence(i))
            Next

            Return shiftedSequence
        End Function

        ''' <summary>
        ''' Converts a list of words to a list of corresponding indices based on the vocabulary.
        ''' </summary>
        ''' <param name="words">The list of words to convert.</param>
        ''' <returns>A list of corresponding indices.</returns>
        Private Function ConvertWordsToIndices(words As List(Of String)) As List(Of Integer)
            Dim indices As New List(Of Integer)

            For Each word As String In words
                If TestVocab.Contains(word) Then
                    indices.Add(TestVocab.IndexOf(word))
                Else
                    indices.Add(TestVocab.IndexOf("UNK")) ' Unknown word
                End If
            Next

            Return indices
        End Function

    End Class
    Public Class TransformerEncoder
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

            Public Function Forward(input As Double()()) As Double()()
                Dim batchSize = input.Length
                Dim output = New Double(batchSize - 1)() {}

                For i = 0 To batchSize - 1
                    Dim inputSequence = input(i)
                    Dim sequenceLength = inputSequence.Length

                    ' Split input sequence into heads
                    Dim inputHeads = SplitSequenceIntoHeads(inputSequence)

                    ' Apply attention to each head
                    Dim headOutputs = New Double(headCount - 1)() {}
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
                        headOutputs(j) = ApplyLinearTransformation(iweightedSum, outputWeight(j))
                    Next

                    ' Concatenate head outputs and apply final linear transformation
                    output(i) = ConcatenateHeadsAndApplyLinearTransformation(headOutputs)
                Next

                Return output
            End Function

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

            Private Function ApplyLinearTransformation(sequence As Double(), weights As Double()) As Double()
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

            Private Function Softmax(scores As Double()) As Double()
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

            Private Function ConcatenateHeadsAndApplyLinearTransformation(headOutputs As Double()()) As Double()
                Dim outputSize = outputWeight(0).Length
                Dim output = New Double(outputSize - 1) {}

                Dim index = 0
                For i = 0 To headOutputs.Length - 1
                    For j = 0 To headOutputs(i).Length - 1
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

            Private Function ComputeVariance(sequence As Double(), mean As Double) As Double
                Dim sum = 0.0
                For Each value In sequence
                    sum += (value - mean) * (value - mean)
                Next

                Return sum / sequence.Length
            End Function

            Private Function Normalize(sequence As Double(), mean As Double, variance As Double) As Double()
                Dim normalized = New Double(sequence.Length - 1) {}

                Dim stdDev = Math.Sqrt(variance + epsilon)
                For i = 0 To sequence.Length - 1
                    normalized(i) = (sequence(i) - mean) / stdDev
                Next

                Return normalized
            End Function
            Public Function Normalize(inputs As List(Of List(Of Double))) As List(Of List(Of Double))
                Dim normalizedOutputs As List(Of List(Of Double)) = New List(Of List(Of Double))(inputs.Count)

                For Each inputVector As List(Of Double) In inputs
                    Dim mean As Double = inputVector.Average()
                    Dim variance As Double = inputVector.Select(Function(x) (x - mean) * (x - mean)).Sum() / hiddenSize
                    Dim stdDev As Double = Math.Sqrt(variance + epsilon)

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

            Public Function Forward(inputs As Double()) As Double()
                Dim hiddenOutputs = ApplyLinearTransformation(inputs, hiddenWeight, hiddenBias)
                Dim hiddenActivated = ReLU(hiddenOutputs)
                Dim outputs = ApplyLinearTransformation(hiddenActivated, outputWeight, outputBias)

                Return outputs
            End Function

            Private Function ApplyLinearTransformation(inputs As Double(), weights As Double()(), bias As Double()) As Double()
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

            Private Function ReLU(inputs As Double()) As Double()
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
        Public Class TransformerEncoderDecoder
            Private encoder As TransformerEncoder
            Private decoder As TransformerEncoder

            Public Sub New(encoder As TransformerEncoder, decoder As TransformerEncoder)
                Me.encoder = encoder
                Me.decoder = decoder
            End Sub

            Public Function SingleForwardPass(inputSequence As List(Of List(Of Double)), targetSequence As List(Of List(Of Double))) As List(Of List(Of Double))
                ' Forward pass through the encoder
                Dim encoderOutput As List(Of List(Of Double)) = encoder.Forward(inputSequence)

                ' Forward pass through the decoder with the encoder output and target sequence
                Dim decoderOutput As List(Of List(Of Double)) = decoder.Forward(encoderOutput)

                ' Return the decoder output
                Return decoderOutput
            End Function


            Public Function BatchForwardPass(inputBatch As List(Of List(Of List(Of Double))), targetBatch As List(Of List(Of List(Of Double)))) As List(Of List(Of List(Of Double)))
                Dim batchOutput As New List(Of List(Of List(Of Double)))

                ' Iterate over each input and target sequence in the batch
                For i As Integer = 0 To inputBatch.Count - 1
                    Dim inputSequence As List(Of List(Of Double)) = inputBatch(i)
                    Dim targetSequence As List(Of List(Of Double)) = targetBatch(i)

                    ' Forward pass through the encoder
                    Dim encoderOutput As List(Of List(Of Double)) = encoder.Forward(inputSequence)

                    ' Forward pass through the decoder with the encoder output and target sequence
                    Dim decoderOutput As List(Of List(Of Double)) = decoder.Forward(encoderOutput)

                    ' Add the decoder output to the batch output
                    batchOutput.Add(decoderOutput)
                Next

                ' Return the batch output
                Return batchOutput
            End Function
        End Class


        Private D_Model As Integer
        Private ReadOnly hiddenSize As Integer
        Private ReadOnly numLayers As Integer
        Private ReadOnly attentionHeads As Integer

        Public Sub New(hiddenSize As Integer, numLayers As Integer, attentionHeads As Integer)
            Me.hiddenSize = hiddenSize
            Me.numLayers = numLayers
            Me.attentionHeads = attentionHeads
        End Sub
        Public Function Forward(inputSequence As List(Of List(Of Double))) As List(Of List(Of Double))
            Dim encodedSequence As List(Of List(Of Double)) = inputSequence
            Me.D_Model = inputSequence(0).Count
            For layerIndex = 0 To numLayers - 1

                ' Apply multi-headed attention
                Dim attention As List(Of List(Of Double)) = MultiHeadAttention.MultiHeadedAttention(encodedSequence, attentionHeads)


                ' Add residual connection and perform layer normalization again
                encodedSequence = MathMult.ConcatenateMatrices(encodedSequence, attention, MathMult.ConcatenationType.Vertical)
                ' Apply layer normalization
                ' Dim layerNorm As New LayerNormalization(hiddenSize)
                'encodedSequence = layerNorm.Normalize(encodedSequence)
            Next

            Return encodedSequence
        End Function
        Public Shared Sub Main()
            ' Create an instance of TransformerEncoder
            Dim model As New TransformerEncoder(hiddenSize:=512, numLayers:=4, attentionHeads:=8)

            ' Prepare input sequence
            Dim inputSequence As New List(Of List(Of Double))
            inputSequence.Add(New List(Of Double) From {0.1, 0.2, 0.3})
            inputSequence.Add(New List(Of Double) From {0.4, 0.5, 0.6})
            inputSequence.Add(New List(Of Double) From {0.7, 0.8, 0.9})

            ' Encode the input sequence
            Dim encodedSequence As List(Of List(Of Double)) = model.Forward(inputSequence)

            ' Print the encoded sequence
            Console.WriteLine("In this example, 
we create an instance of TransformerEncoder 
with a hidden size of 512, 4 layers, and 8 attention heads. 
 we iterate over the encoded sequence and print each token.")
            For Each token In encodedSequence
                Console.WriteLine(String.Join(", ", token))
            Next

            ' Example usage
            Dim predictions As List(Of List(Of Double)) = CreateRandomList(3, 5) ' Mock predictions
            Dim targets As List(Of List(Of Double)) = CreateRandomList(3, 5) ' Mock targets

            Dim loss As Double = ComputeCrossEntropyLoss(predictions, targets)
            Console.WriteLine("Cross Entropy Loss: " & loss)
            Dim Gradients As List(Of List(Of Double)) = ComputeGradients(predictions, targets)
            Console.WriteLine("Cross Entropy Loss: " & loss)

        End Sub
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
        Public Sub PrintMatrix(matrix As List(Of List(Of Double)))
            For Each row In matrix
                For Each value In row
                    Console.Write(value.ToString("0.0000") & " ")
                Next
                Console.WriteLine()
            Next
        End Sub
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

    End Class

End Namespace
