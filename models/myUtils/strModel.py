import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import pandas as pd

def concatTxt(originalTxt, newTxt, headTail=True):
    """
    :param originalTxt:
    :param newTxt:
    :param targetCode:
    :return:
    """
    targetCodes = {}
    for fileName, code in originalTxt.copy().items():
        if fileName != newTxt:
            if headTail:
                targetCodes[fileName] = originalTxt[newTxt] + '\n' + originalTxt[fileName]
            else:
                targetCodes[fileName] = originalTxt[fileName] + '\n' + originalTxt[newTxt]
    return targetCodes

def replaceAllTxt(txt, replacedTable):
    """
    txt: str
    replacedTable: dict, eg: {'\$AT__': '@@', '\$EQ__': '@=', ...}
    """
    for replacedObj, replacedValue in replacedTable.items():
        # txt = txt.replace(replacedObj, replacedValue)
        txt = re.sub(replacedObj, replacedValue, txt, count=0)
    return txt

# if pattern exist, then return True
def patternsExisted(txt: str, patterns: list):
    existed = False
    for pattern in patterns:
        existed = existed | bool(re.search(pattern, txt))
    return existed

def getStemmText(txt, language='english'):
    """
    eg:
    13,000 people receive #wildfires evacuation ... -> people receive wildfires evacuation ...
    Just got sent this photo from Ruby #Alaska as ... -> got sent photo ruby alaska ...
    """
    stemmer = nltk.SnowballStemmer(language)
    cleanTxt = ' '.join([stemmer.stem(word) for word in txt.split(' ')])
    return cleanTxt

def removeStopwords(txt, specialStopWords=None, language='english'):
    """
    remove the stop word, eg: i, me, my, we, ...
    :param txt: str
    :param specialStopWords: [], special character
    :return:
    """
    if specialStopWords==None:
        specialStopWords = []
    stop_words = stopwords.words(language)
    stop_words = stop_words + specialStopWords
    words = txt.split(' ')
    words = [word for word in words if word not in stop_words]
    cleanTxt = ' '.join(words)
    return cleanTxt

def downloadNltk():
    """
    Download the NLTK corpus

    """
    nltk.download('stopwords')

# df = pd.read_csv("C:/Users/Chris/projects/221227_mt5Mvc/docs/datas/nlp-getting-started/train.csv")
# df['clean_text'] = df['text'].apply(getStemmText)
# df['clean_text'] = df['clean_text'].apply(removeStopwords)
#
# vectorizer = TfidfVectorizer()
# X_train = vectorizer.fit_transform(df['clean_text'])
# print(df)