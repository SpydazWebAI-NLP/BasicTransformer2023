Namespace Basic_NLP

    Namespace PositionalEncoding
        ''' <summary>
        ''' Encoding:
        ''' EncodeTokenStr: Encodes a String token And returns its positional embedding As a list Of doubles.
        '''    EncodeTokenEmbedding: Encodes a token embedding (list Of doubles) And returns its positional embedding As a list Of doubles.
        '''    EncodeSentenceStr: Encodes a list Of String tokens And returns their positional embeddings As a list Of lists Of doubles.
        '''    EncodeSentenceEmbedding: Encodes a list Of token embeddings And returns their positional embeddings As a list Of lists Of doubles.
        '''Decoding:
        '''DecodeTokenStr: Decodes a positional embedding (list Of doubles) And returns the corresponding String token.
        '''    DecodeTokenEmbedding: Decodes a positional embedding (list Of doubles) And returns the corresponding token embedding As a list Of doubles.
        '''    DecodeSentenceStr: Decodes a list Of positional embeddings And returns the corresponding String tokens As a list Of strings.
        '''    DecodeSentenceEmbedding: Decodes a list Of positional embeddings And returns the corresponding token embeddings As a list Of lists Of doubles.
        '''     </summary>
        Public Class PositionalEncoderDecoder
            Private PositionalEmbedding As New List(Of List(Of Double))
            Private Vocabulary As New List(Of String)

            Public Sub New(ByRef Dmodel As Integer, MaxSeqLength As Integer, vocabulary As List(Of String))
                ' Create Learnable Positional Embedding Layer
                PositionalEmbedding = CreatePositionalEmbedding(Dmodel, MaxSeqLength)
                ' Set Reference Vocabulary
                Me.Vocabulary = vocabulary
            End Sub

            ' Encode
            Public Function EncodeTokenStr(ByRef nToken As String, position As Integer) As List(Of Double)
                Dim positionID As Integer = GetTokenIndex(nToken)
                Dim ipositionalEmbedding As List(Of Double) = PositionalEmbedding(position)
                Dim encodingVector As List(Of Double) = If(positionID <> -1, ipositionalEmbedding, New List(Of Double))
                Return encodingVector
            End Function
            Public Function EncodeTokenStr(ByRef nToken As String) As List(Of Double)
                Dim positionID As Integer = GetTokenIndex(nToken)
                Return If(positionID <> -1, PositionalEmbedding(positionID), New List(Of Double))
            End Function

            Public Function EncodeSentenceStr(ByRef Sentence As List(Of String)) As List(Of List(Of Double))
                Dim EncodedSentence As New List(Of List(Of Double))
                For i = 0 To Sentence.Count - 1
                    Dim encodingVector As List(Of Double) = EncodeTokenStr(Sentence(i), i)
                    EncodedSentence.Add(encodingVector)
                Next
                Return EncodedSentence
            End Function

            ' Decode
            Public Function DecodeSentenceStr(ByRef Sentence As List(Of List(Of Double))) As List(Of String)
                Dim DecodedSentence As New List(Of String)
                For Each tokenEncoding In Sentence
                    Dim decodedToken As String = DecodeTokenStr(tokenEncoding)
                    DecodedSentence.Add(decodedToken)
                Next
                Return DecodedSentence
            End Function
            ''' <summary>
            ''' Used For String Tokens
            ''' </summary>
            ''' <param name="PositionalEmbeddingVector"></param>
            ''' <returns>String Token</returns>
            Public Function DecodeTokenStr(ByRef PositionalEmbeddingVector As List(Of Double)) As String
                Dim positionID As Integer = GetPositionID(PositionalEmbeddingVector)
                Return If(positionID <> -1, Vocabulary(positionID), "")
            End Function
            ' Create Learnable Positional Embedding Layer
            Private Function CreatePositionalEmbedding(Dmodel As Integer, MaxSeqLength As Integer) As List(Of List(Of Double))
                Dim positionalEmbeddings As New List(Of List(Of Double))
                Dim rnd As New Random()

                For pos As Integer = 0 To MaxSeqLength - 1
                    Dim embeddingRow As List(Of Double) = New List(Of Double)

                    For i As Integer = 0 To Dmodel - 1
                        ' Initialize the positional embeddings with random values
                        embeddingRow.Add(rnd.NextDouble())
                    Next

                    positionalEmbeddings.Add(embeddingRow)
                Next

                Return positionalEmbeddings
            End Function

            Private Function GetTokenIndex(PositionalEncoding As List(Of Double)) As Integer

                For i As Integer = 0 To PositionalEmbedding.Count - 1
                    If PositionalEncoding.SequenceEqual(PositionalEmbedding(i)) Then
                        Return i
                    End If
                Next

                Return -1 ' Token not found
            End Function
            Private Function GetTokenIndex(token As String) As Integer

                Return Vocabulary.IndexOf(token)
            End Function
            Private Function GetPositionID(PositionalEmbeddingVector As List(Of Double)) As Integer
                For i As Integer = 0 To PositionalEmbedding.Count - 1
                    If PositionalEmbeddingVector.SequenceEqual(PositionalEmbedding(i)) Then
                        Return i
                    End If
                Next

                Return -1 ' Position ID not found
            End Function

        End Class

    End Namespace
End Namespace