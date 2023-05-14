import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def train_test_split(): 
    # TODO: output X_train, X_test, y_train, y_test
    raise NotImplementedError

def transform(): 
    # TODO: TF-IDF, otherwise, data leakage
    raise NotImplementedError

def grid_search_params():
    # TODO: find best params
    raise NotImplementedError

def save_best_params():
    # TODO: save best params, either as pickle file or dictionary
    raise NotImplementedError