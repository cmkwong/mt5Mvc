import os
import shutil
import inspect
from mt5Mvc.models.myUtils import listModel

def getFileList(pathDir, reverse=False):
    fileList = []
    filenames = os.listdir(pathDir)
    for fileName in filenames:
        if fileName[0] != '~': # discard the temp file
            fileList.append(fileName)
    fileList = sorted(fileList, reverse=reverse)
    return fileList

# delete folder
def delFolder(mainPath, folderName):
    folderPath = os.path.join(mainPath, folderName)
    if os.path.exists(folderPath):
        # Delete Folder code
        shutil.rmtree(folderPath)


# clear the files inside of folder
def clearFiles(pathDir, pattern=None):
    """
    pattern None means clear all files in the pathDir
    """
    files = getFileList(pathDir)
    if pattern:
        files = listModel.filterList(files, pattern)
    for file in files:
        os.remove(os.path.join(pathDir, file))
        print("The file {} has been removed.".format(file))

def createDir(mainPath, dirName, gitKeep=False, readme=None):
    """
    Create the folder if not exist
    :gitKeep: Create directory with .gitKeep
    :readme: Create directory with readme.txt
    """
    fullpath = os.path.join(mainPath, dirName)
    if not os.path.isdir(fullpath):
        os.mkdir(fullpath)
    if readme:
        with open(os.path.join(mainPath, 'readme.txt'), 'a', encoding='utf-8') as f:
            f.write(readme)
    if gitKeep:
        createFile(os.path.join(mainPath, dirName), '.gitkeep')
    return fullpath

def createFile(mainPath, fileName, txt=None):
    with open(os.path.join(mainPath, fileName), 'a', encoding='utf-8') as f:
        if txt: f.write(txt)

def read_text(main_path, file_name):
    with open(os.path.join(main_path, file_name), 'r', encoding='utf-8') as f:
        txt = f.read()
    return txt

def readAllTxtFiles(fileDir, outFormat=dict):
    """
    :param fileDir: str
    :return: {}
    """
    output = outFormat() # define the init data type
    for curPath, directories, files in os.walk(fileDir): # deep walk
        for file in files:
            with open(os.path.join(curPath, file), 'r', encoding='UTF-8') as f:
                if outFormat == dict:
                    output[file] = f.read()
                elif outFormat == str:
                    output += f.read() + '\n'
    return output

def writeAllTxtFiles(main_path, texts):
    """
    :param texts: dic
    :param path: str
    :return:
    """
    for fileName, code in texts.items():
        if len(fileName) > 0 and fileName[0] != '_':
            with open(os.path.join(main_path, fileName), 'w', encoding='UTF-8') as f:
                f.write(code)
            print("Written {}".format(fileName))

def getParentFolderName(classObj):
    pathStr = inspect.getfile(classObj)
    parentFolder = os.path.basename(os.path.split(pathStr)[0])
    return parentFolder