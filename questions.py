import nltk
import sys
import math
import os
import string

# nltk.download('stopwords')

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    contents = dict()

    for filename in os.listdir(directory):
        txtfile = os.path.join(directory, filename)
        if os.path.isfile(txtfile) and filename.endswith(".txt"):
            with open(txtfile, "r", encoding='utf8') as file:
                contents[filename] = file.read()

    return contents


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    tokens = nltk.word_tokenize(document)

    words = []
    # Add words that are not punctuation and English stopwords
    for word in tokens:
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            words.append(word)
    
    return words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    contained = dict()
    idfs = dict()

    # Append number of documents that have a word
    for filename in documents:
        seen = set()
        for word in documents[filename]:
            if word not in seen:
                seen.add(word)
                if word in contained:
                    contained[word] += 1
                else:
                    contained[word] = 1
    
    # Calculate the idf of each word
    for word, count in contained.items():
        idf = math.log(len(documents) / count)
        idfs[word] = idf

    return idfs
        

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    # Initualize each worth with tfidf = 0
    tfidf = dict.fromkeys(files.keys(),0)

    for txtfile in tfidf:
        for word in query:
            totalWords = files[txtfile]
            # Record the tfidf values
            if word in totalWords:
                tf = totalWords.count(word) / len(totalWords)
                idf = idfs[word]
                tfidf[txtfile] = tf * idf
    
    rank = []

    # Get n files with highest tfidf value
    for i in range(n):
        maximum = max(tfidf, key=tfidf.get)
        rank.append(maximum)
        del tfidf[maximum]

    return rank


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    matchedList = []

    # Record the idf and qtd(query term density) of each sentence
    for sentence, words in sentences.items():
        idf = 0
        qtd = 0
        for word in query:
            if word in words:
                idf += idfs[word]
                qtd += words.count(word) / len(words)
        
        # Append a tuple of matched sentence with its idf and qtd
        matchedList.append((sentence, idf, qtd))

    topSentences = []

    for i in range(n):
        # Sort and get the sentence with highest idf and qtd
        maxSentence = max(matchedList,key=lambda x:(x[1],x[2]))
        topSentences.append(maxSentence[0])
        matchedList.remove(maxSentence)

    return topSentences
            

        


if __name__ == "__main__":
    main()
