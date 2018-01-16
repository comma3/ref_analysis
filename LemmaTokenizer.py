from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    """
    Implemenation of a tokenizer/lemmatizer from sklearn's documentation.
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
