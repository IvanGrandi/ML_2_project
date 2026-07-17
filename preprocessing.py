import re
import pandas as pd
import html
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

try:
    nlp = spacy.load("fr_core_news_sm", disable=["ner", "parser"])
except OSError:
    raise OSError(
        "Le modèle SpaCy fr_core_news_sm n'est pas installé.\n"
        "Exécutez d'abord : python -m spacy download fr_core_news_sm"
    )

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", quiet=True)

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# Convert text to lowercase
def lowerText(text):
    return str.lower(text)


# Decode HTML entities : Convert &amp; as &, &lt; as <, etc.
def decodeHTML(text):
    if not text or pd.isna(text):
        return ""
    return html.unescape(str(text))

# Remove punctuation from text : We replace punctuation with spaces
def removePonctuation(text):
    text = re.sub(r"[^\w\s]", " ", text)

    # We replace multiple spaces with a single space
    text = re.sub(r"\s+", " ", text).strip()

    return text

# Tokenize text into words
# def tokenize(text):
#    return nltk.word_tokenize(text)

# spaCy Tokenizer which is more accurate than NLTK's tokenizer, especially for French text
def tokenize(text):
    doc = nlp.make_doc(text)
    return [token.text for token in doc]

# Remove stop words from list of tokens
def removeStopWords(tokens):
    stop_words = set(stopwords.words("french"))
    tokens = [w for w in tokens if w not in stop_words]
    return tokens

# Remove @mentions from text
def removeMentions(text):
    return re.sub(r"@[A-Za-z0-9_]+", "", text)


# Remove http/https links from text
def removeLinks(text):
    return re.sub(r"https?://[A-Za-z0-9./]+", "", text)


# Remove hashtags from text (just the # symbol)
def removeHashtags(text):
    return re.sub(r"#(\w+)", r"\1", text)


# Lemmatize tokens
def lemmatizeTokens(tokens):
    text_recon = " ".join(tokens)
    doc = nlp(text_recon)
    lemmas = [token.lemma_ for token in doc if token.lemma_ != "-PRON-"]
    return lemmas


# Stem tokens
def stemTokens(tokens):
    stemmer = SnowballStemmer("french")
    return [stemmer.stem(w) for w in tokens]

# Get cleaning steps for visualization : Apply each step and store the result for display
def get_cleaning_steps(text):

    steps = {}

    # Step 1 : Original
    current = text
    steps["Original"] = current


    # Step 2 : Decoding HTML
    current = decodeHTML(current)
    steps["Decoded HTML"] = current

    # Step 3 : Cleaning (Mentions + Links)
    current = removeLinks(current)
    #current = removeMentions(current)
    #current = removeHashtags(current)
    steps["Cleaned Text"] = current
   
    # Step 4 : Lowercase
    current = lowerText(current)
    steps["Lowercase"] = current

    # Step 5 : Punctuation
    current = removePonctuation(current)
    steps["No Punctuation"] = current

    # Step 6 : Tokenization
    tokens = tokenize(current)
    steps["Tokenized"] = str(tokens)  # We store as string for display

    # Step 7 : Stop Words
    tokens = removeStopWords(tokens)
    steps["No StopWords"] = tokens

    # Step 8 : Stemming
    steps["Stemmed"] = str(stemTokens(tokens))

    # Step 9 : Lemmatization
    tokens = lemmatizeTokens(tokens)
    steps["Lemmatized"] = str(tokens)

    return steps

# ... (tes autres imports et fonctions restent inchangés)

# Clean input: replace newlines with spaces and strip outer whitespaces
def clean_input(text):
    if not text or pd.isna(text):
        return ""
    text = str(text).replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


