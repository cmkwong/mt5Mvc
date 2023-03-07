import os
import threading

# get the parent path
def getTargetPath(targetName):
    currentPath = os.getcwd()
    while True:
        if (os.path.basename(currentPath) == targetName):
            return currentPath
        base = os.path.basename(os.path.abspath(os.path.join(currentPath, os.pardir)))
        currentPath = os.path.dirname(currentPath)
        if base == targetName:
            return currentPath
        if len(base) == 0:
            raise Exception("No such target name found. ")

# threading to running a function
def thread_start(fn, *args):
    thread = threading.Thread(target=fn, args=args)
    thread.start()

# find required path
def find_required_path(path, target):
    while(True):
        head, tail = os.path.split(path)
        if (len(tail) == 0): # cannot split anymore
            return head
        if tail == target:
            return path
        path = head