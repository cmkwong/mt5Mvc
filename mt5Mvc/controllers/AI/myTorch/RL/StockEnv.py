import enum

class Actions(enum.Enum):
    Skip = 0
    Buy = 1
    Close = 2

class StockEnv:
    def __init__(self, Prices, windows: int=50):
        self.Prices = Prices
        self.index = windows
        self.windows = windows

    def reset(self):
        self.index = self.windows

    def step(self):
        self.Prices = None
