from models.myUtils import sysModel

def command_check(target_commands: list):
    def decorator(fn):
        def wrapper(**kwargs):
            if target_commands in kwargs['command']:
                return fn(**kwargs)
            return False
        return wrapper
    return decorator

# class Base_Handler:
#     def __init__(self):
#         self.command = None
#
#     def run(self, command):
#         self.command = command
#         sysModel.run_all_func(self, suffix='__exec')

