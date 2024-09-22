import os
import inspect
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

# find required path
def find_required_path(path, target):
    while(True):
        head, tail = os.path.split(path)
        if (len(tail) == 0): # cannot split anymore
            return head
        if tail == target:
            return path
        path = head

# threading to running a function
def thread_start(fn, *args):
    thread = threading.Thread(target=fn, args=args)
    thread.start()

# run all functions from class instance
def run_all_func(inst, prefix=None, suffix=None):
    """
    :param inst: class instance
    :param prefix: condition that only these meet with prefix
    """
    attrs = (getattr(inst, name) for name in dir(inst))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        try:
            if (prefix == None and suffix == None) or \
                    (prefix and suffix == None and method.__name__.startswith(prefix)) or \
                    (prefix == None and suffix and method.__name__.endswith(suffix)) or \
                    (prefix and suffix and method.__name__.startswith(prefix) and method.__name__.endswith(suffix)):
                method()
        except TypeError:
            # Can't handle methods with required arguments.
            print(TypeError)