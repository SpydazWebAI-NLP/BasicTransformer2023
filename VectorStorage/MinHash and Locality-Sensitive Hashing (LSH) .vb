Namespace MinHashAndLSH
    Module MainModule
        Sub Main()
            ' Create an LSHIndex object with 3 hash tables and 2 hash functions per table
            Dim lshIndex As New LSHIndex(numHashTables:=3, numHashFunctionsPerTable:=2)

            ' Create some sample articles
            Dim article1 As New Document("The Art of Baking: Mastering the Perfect Chocolate Cake", 0)
            Dim article2 As New Document("Exploring Exotic Cuisines: A Culinary Adventure in Southeast Asia", 1)
            Dim article3 As New Document("Nutrition for Optimal Brain Health: Foods that Boost Cognitive Function", 2)
            Dim article4 As New Document("The Rise of Artificial Intelligence: A Game-Changer in the Tech World", 3)
            Dim article5 As New Document("Introduction to Quantum Computing: Unraveling the Power of Qubits", 4)

            ' Add the articles to the LSH index
            lshIndex.AddDocument(article1)
            lshIndex.AddDocument(article2)
            lshIndex.AddDocument(article3)
            lshIndex.AddDocument(article4)
            lshIndex.AddDocument(article5)

            ' Create a query article
            Dim queryArticle As New Document("Delicious Desserts: A Journey Through Sweet Delights", -1)

            ' Find similar articles using LSH
            Dim similarArticles As List(Of Document) = lshIndex.FindSimilarDocuments(queryArticle)

            ' Display the results
            Console.WriteLine("Query Article: " & queryArticle.Content)
            If similarArticles.Count = 0 Then
                Console.WriteLine("No similar articles found.")
            Else
                Console.WriteLine("Similar Articles:")
                For Each article As Document In similarArticles
                    Console.WriteLine(article.Content)
                Next
            End If

            Console.ReadLine()
        End Sub
    End Module
    Public Class LSHIndex
        Private HashTables As List(Of Dictionary(Of Integer, List(Of Document)))
        Private NumHashTables As Integer
        Private NumHashFunctionsPerTable As Integer

        Public Sub New(ByVal numHashTables As Integer, ByVal numHashFunctionsPerTable As Integer)
            ' Initialize the LSH index with the specified number of hash tables and hash functions per table
            Me.NumHashTables = numHashTables
            Me.NumHashFunctionsPerTable = numHashFunctionsPerTable
            HashTables = New List(Of Dictionary(Of Integer, List(Of Document)))(numHashTables)

            ' Create empty hash tables
            For i As Integer = 0 To numHashTables - 1
                HashTables.Add(New Dictionary(Of Integer, List(Of Document))())
            Next
        End Sub

        Private Function ComputeHash(ByVal content As String, ByVal hashFunctionIndex As Integer) As Integer
            ' Improved hash function that uses prime numbers to reduce collisions.
            ' This hash function is for demonstration purposes and can be further optimized for production.
            Dim p As Integer = 31 ' Prime number
            Dim hash As Integer = 0
            For Each ch As Char In content
                hash = (hash * p + Asc(ch)) Mod Integer.MaxValue
            Next
            Return hash
        End Function

        Public Sub AddDocument(ByVal document As Document)
            ' For each hash table, apply multiple hash functions to the document and insert it into the appropriate bucket.
            For tableIndex As Integer = 0 To NumHashTables - 1
                For i As Integer = 0 To NumHashFunctionsPerTable - 1
                    Dim hashCode As Integer = ComputeHash(document.Content, i)
                    If Not HashTables(tableIndex).ContainsKey(hashCode) Then
                        ' If the bucket for the hash code doesn't exist, create it and associate it with an empty list of documents.
                        HashTables(tableIndex)(hashCode) = New List(Of Document)()
                    End If
                    ' Add the document to the bucket associated with the hash code.
                    HashTables(tableIndex)(hashCode).Add(document)
                Next
            Next
        End Sub

        Private Function GenerateShingles(ByVal content As String, ByVal shingleSize As Integer) As List(Of String)
            ' Helper function to generate shingles (substrings) of the given size from the content.
            Dim shingles As New List(Of String)()
            If content.Length < shingleSize Then
                shingles.Add(content)
                Return shingles
            End If
            For i As Integer = 0 To content.Length - shingleSize
                shingles.Add(content.Substring(i, shingleSize))
            Next
            Return shingles
        End Function

        Private Function ComputeSimilarity(ByVal content1 As String, ByVal content2 As String) As Double
            ' Improved Jaccard similarity function that uses sets of shingles for more accurate results.
            ' For simplicity, let's use 3-character shingles.
            Dim set1 As New HashSet(Of String)(GenerateShingles(content1, 3))
            Dim set2 As New HashSet(Of String)(GenerateShingles(content2, 3))

            Dim intersectionCount As Integer = set1.Intersect(set2).Count()
            Dim unionCount As Integer = set1.Count + set2.Count - intersectionCount

            Return CDbl(intersectionCount) / CDbl(unionCount)
        End Function

        Public Function FindSimilarDocuments(ByVal queryDocument As Document) As List(Of Document)
            Dim similarDocuments As New List(Of Document)()

            ' For the query document, compute hash codes for each hash table and look for similar documents.
            For tableIndex As Integer = 0 To NumHashTables - 1
                For i As Integer = 0 To NumHashFunctionsPerTable - 1
                    Dim hashCode As Integer = ComputeHash(queryDocument.Content, i)
                    Dim similarDocs As List(Of Document) = FindSimilarDocumentsInHashTable(hashCode, tableIndex)
                    similarDocuments.AddRange(similarDocs)
                Next
            Next

            ' Remove duplicates from the list of similar documents.
            Dim uniqueSimilarDocs As New List(Of Document)(New HashSet(Of Document)(similarDocuments))

            ' Sort the similar documents based on their similarity to the query document.
            uniqueSimilarDocs.Sort(Function(doc1, doc2) ComputeSimilarity(queryDocument.Content, doc2.Content).CompareTo(ComputeSimilarity(queryDocument.Content, doc1.Content)))

            Return uniqueSimilarDocs
        End Function

        Private Function FindSimilarDocumentsInHashTable(ByVal queryDocHashCode As Integer, ByVal tableIndex As Integer) As List(Of Document)
            ' Given the hash code of the query document and a hash table index, find all documents in the same bucket.
            Dim similarDocs As New List(Of Document)()
            If HashTables(tableIndex).ContainsKey(queryDocHashCode) Then
                ' If the bucket for the hash code exists, retrieve the list of documents in the bucket.
                similarDocs.AddRange(HashTables(tableIndex)(queryDocHashCode))
            End If
            Return similarDocs
        End Function
    End Class
    Public Class MinHashAndLSHInterface
        Private MinHashIndex As MinHashIndex

        Public Sub New(ByVal numHashFunctions As Integer)
            MinHashIndex = New MinHashIndex(numHashFunctions)
        End Sub

        Public Sub AddDocument(ByVal content As String, ByVal index As Integer)
            Dim document As New Document(content, index)
            MinHashIndex.AddDocument(document)
        End Sub

        Public Function FindSimilarDocuments(ByVal queryContent As String) As List(Of Document)
            Dim queryDocument As New Document(queryContent, -1)
            Dim similarDocuments As List(Of Document) = MinHashIndex.FindSimilarDocuments(queryDocument)
            Return similarDocuments
        End Function
    End Class
    Public Class Document
        Public Property Index As Integer
        Public Property Content As String
        Public Property Elements As List(Of String)

        Public Sub New(ByVal content As String, ByVal index As Integer)
            Me.Content = content
            Me.Index = index
            ' Split the document content into elements (words) for MinHash.
            Me.Elements = content.Split({" "}, StringSplitOptions.RemoveEmptyEntries).ToList()
        End Sub
    End Class
    Public Class MinHashVectorDatabase
        Private MinHashIndex As MinHashIndex

        Public Sub New(ByVal numHashFunctions As Integer)
            MinHashIndex = New MinHashIndex(numHashFunctions)
        End Sub

        Public Sub AddDocument(ByVal content As String, ByVal index As Integer)
            Dim document As New Document(content, index)
            MinHashIndex.AddDocument(document)
        End Sub

        Public Function FindSimilarDocuments(ByVal queryContent As String, ByVal threshold As Double) As List(Of Document)
            Dim queryDocument As New Document(queryContent, -1)
            Dim similarDocuments As List(Of Document) = MinHashIndex.FindSimilarDocuments(queryDocument)
            Return similarDocuments
        End Function
    End Class
    Public Class MinHashIndex
        Private NumHashFunctions As Integer
        Private SignatureMatrix As List(Of List(Of Integer))
        Private Buckets As Dictionary(Of Integer, List(Of Document))

        Public Sub New(ByVal numHashFunctions As Integer)
            Me.NumHashFunctions = numHashFunctions
            SignatureMatrix = New List(Of List(Of Integer))()
            Buckets = New Dictionary(Of Integer, List(Of Document))()
        End Sub

        Private Function ComputeHash(ByVal element As String, ByVal hashFunctionIndex As Integer) As Integer
            ' Use a simple hashing algorithm for demonstration purposes.
            Dim p As Integer = 31 ' Prime number
            Dim hash As Integer = 0
            For Each ch As Char In element
                hash = (hash * p + Asc(ch)) Mod Integer.MaxValue
            Next
            Return hash
        End Function

        Public Sub AddDocument(ByVal document As Document)
            ' Generate the signature matrix for the given document using the hash functions.
            For i As Integer = 0 To NumHashFunctions - 1
                Dim minHash As Integer = Integer.MaxValue
                For Each element As String In document.Elements
                    Dim hashCode As Integer = ComputeHash(element, i)
                    minHash = Math.Min(minHash, hashCode)
                Next
                If SignatureMatrix.Count <= i Then
                    SignatureMatrix.Add(New List(Of Integer)())
                End If
                SignatureMatrix(i).Add(minHash)
            Next

            ' Add the document to the appropriate bucket for LSH.
            Dim bucketKey As Integer = SignatureMatrix(NumHashFunctions - 1).Last()
            If Not Buckets.ContainsKey(bucketKey) Then
                Buckets(bucketKey) = New List(Of Document)()
            End If
            Buckets(bucketKey).Add(document)
        End Sub

        Private Function EstimateSimilarity(ByVal signatureMatrix1 As List(Of Integer), ByVal signatureMatrix2 As List(Of Integer)) As Double
            Dim matchingRows As Integer = 0
            For i As Integer = 0 To NumHashFunctions - 1
                If signatureMatrix1(i) = signatureMatrix2(i) Then
                    matchingRows += 1
                End If
            Next

            Return CDbl(matchingRows) / CDbl(NumHashFunctions)
        End Function

        Public Function FindSimilarDocuments(ByVal queryDocument As Document) As List(Of Document)
            ' Generate the signature matrix for the query document using the hash functions.
            Dim querySignature As New List(Of Integer)()
            For i As Integer = 0 To NumHashFunctions - 1
                Dim minHash As Integer = Integer.MaxValue
                For Each element As String In queryDocument.Elements
                    Dim hashCode As Integer = ComputeHash(element, i)
                    minHash = Math.Min(minHash, hashCode)
                Next
                querySignature.Add(minHash)
            Next

            ' Find candidate similar documents using Locality-Sensitive Hashing (LSH).
            Dim candidateDocuments As New List(Of Document)()
            Dim bucketKey As Integer = querySignature(NumHashFunctions - 1)
            If Buckets.ContainsKey(bucketKey) Then
                candidateDocuments.AddRange(Buckets(bucketKey))
            End If

            ' Refine the candidate documents using MinHash similarity estimation.
            Dim similarDocuments As New List(Of Document)()
            For Each candidateDoc As Document In candidateDocuments
                Dim similarity As Double = EstimateSimilarity(querySignature, SignatureMatrix(candidateDoc.Index))
                If similarity >= 0.5 Then ' Adjust the threshold as needed.
                    similarDocuments.Add(candidateDoc)
                End If
            Next

            Return similarDocuments
        End Function
    End Class
End Namespace
