from mt5Mvc.models.myUtils import listModel

def enter(placeholder='Input: '):
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    user_input = input(placeholder)  # waiting user input
    return user_input


def askNum(placeholder="Please enter a number: ", outType=int):
    """
    change the type for user input, float, int etc
    """
    print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
    usr_input = input(placeholder)
    if not usr_input.isnumeric():
        print("Wrong input. \nPlease input again.\n")
        return None
    usr_input = outType(usr_input)
    return usr_input

# ask for confirm (Y / N)
def askConfirm(question=''):
    if question: print(question)
    placeholder = 'Input [y]es to confirm OR others to cancel: '
    confirm_input = input(placeholder)
    if confirm_input == 'y' or confirm_input == "yes":
        return True
    else:
        return False

# ask user for selection from a {txt: callback}
def askSelection(options: list, prefixTxt:str = 'Please Select'):
    placeholder = f"{listModel.optionsTxt(options)}\n{prefixTxt}: "
    userInput = askNum(placeholder)
    return userInput

# def askDate(placeholder='Please input the date: ', defaultDate='', dateFormat="%Y-%m-%d %H:%M:%S"):
#     """
#     Ask for the date: (2022, 10, 30, 22:21)
#     return: tuple (2022, 1, 20, 5, 45)
#     """
#     print("~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*~*")
#     usr_input = input(f"{placeholder} (default: {defaultDate}) ({dateFormat})\nInput: ")
#     requiredDate = usr_input
#     # if user input empty, set into default date
#     if not usr_input:
#         now = datetime.now()
#         requiredDate = (now.year, now.month, now.day, now.hour, now.minute, now.second)
#         if defaultDate:
#             requiredDate = defaultDate
#
#     return timeModel.getTimeT(requiredDate, dateFormat, dateFormat, True)
