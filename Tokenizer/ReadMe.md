# Tokenizer (VB.NET)

The Tokenizer is a versatile text processing library written in Visual Basic (VB.NET). It provides functionalities for tokenizing text into words, sentences, characters, and n-grams. The library is designed to be flexible, customizable, and easy to integrate into your VB.NET projects.

## Features

- Tokenize text into words
- Tokenize text into sentences
- Tokenize text into character-level tokens
- Generate n-grams from text
- Build vocabulary from tokenized text
- Normalize input text (lowercase, remove punctuation, etc.)
- Remove stop words from tokenized text
- Customize tokenization behavior through various options.

## Usage

1. Initialize the Tokenizer object:

```vb
Dim tokenizer As New Tokenizer()
'Tokenize text:

Dim text As String = "Hello, world! This is a sample sentence."
Dim words As List(Of String) = tokenizer.TokenizeToWords(text)
Dim sentences As List(Of String) = tokenizer.TokenizeToSentence(text)
Dim characters As List(Of String) = tokenizer.TokenizeToCharacter(text)

'Generate Ngrams

Dim sentence As String = "The quick brown fox jumps over the lazy dog."
Dim ngramSize As Integer = 3
Dim ngrams As List(Of Tokenizer.Token) = tokenizer.CreateNgrams(sentence, ngramSize)

'Build Vocabulary
Dim words As List(Of String) = tokenizer.TokenizeToWords(text)
tokenizer.UpdateVocabulary(words)
Dim vocabulary As Dictionary(Of String, Integer) = tokenizer.VocabularyWithFrequency

'Stop Words
tokenizer.StopWords = New List(Of String) From {"is", "a", "the"} ' Set custom stop words
tokenizer.StopWordRemovalEnabled = True ' Enable stop word removal
tokenizer.NGramSize = 2 ' Set n-gram size

'Tokenize and Build Vocabulary
Dim tokenizer As New Tokenizer()
Dim text As String = "This is a sample text. It contains multiple sentences."
Dim words As List(Of String) = tokenizer.TokenizeToWords(text)

tokenizer.UpdateVocabulary(words)
Dim vocabulary As Dictionary(Of String, Integer) = tokenizer.VocabularyWithFrequency

For Each token As KeyValuePair(Of String, Integer) In vocabulary
    Console.WriteLine("Token: " & token.Key & ", Frequency: " & token.Value)
Next


'Tokenize to Chars
Dim tokenizer As New Tokenizer()
Dim text As String = "Hello, world!"
Dim characters As List(Of String) = tokenizer.TokenizeToCharacter(text)

For Each character As String In characters
    Console.WriteLine(character)
Next

