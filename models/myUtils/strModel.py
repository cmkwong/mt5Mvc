import re

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
    replacedTable: dict
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
