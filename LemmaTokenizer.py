from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class LemmaTokenizer(object):
    """
    Modification of a tokenizer/lemmatizer from sklearn's documentation.

    Team nicknames_dict must be ordered dict with longer names first (e.g., tar heels should checked before heels).
    """
    def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.team_dict = team_dict
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]
