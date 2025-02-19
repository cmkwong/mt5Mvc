import enum
from typing import Any, Tuple
import numpy as np

class Actions(enum.Enum):
    """Define possible actions for the environment"""
    Skip = 0  # Do nothing
    Buy = 1   # Open a position
    Close = 2 # Close current position

class StockEnv:
    """Template for stock trading environment
    
    Attributes:
        prices (np.ndarray): Historical price data
        window_size (int): Number of past observations to use as state
        current_step (int): Current position in the price data
        position (int): Current position (0 = no position, 1 = long)
    """
    
    def __init__(self, prices: np.ndarray, window_size: int = 50):
        """Initialize environment
        
        Args:
            prices (np.ndarray): Historical price data (OHLC format)
            window_size (int): Number of past observations to use as state
        """
        self.prices = prices
        self.window_size = window_size
        self.current_step = window_size
        self.position = 0  # 0 = no position, 1 = long
        
    def reset(self) -> np.ndarray:
        """Reset environment to initial state
        
        Returns:
            np.ndarray: Initial state observation
        """
        self.current_step = self.window_size
        self.position = 0
        return self._get_observation()
        
    def step(self, action: Actions) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment
        
        Args:
            action (Actions): Action to take
            
        Returns:
            Tuple containing:
                - state (np.ndarray): New state observation
                - reward (float): Reward from taking action
                - done (bool): Whether episode is finished
                - info (dict): Additional information
        """
        # TODO: Implement action logic
        if action == Actions.Buy:
            self.position = 1
        elif action == Actions.Close:
            self.position = 0
            
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if episode is done
        done = self.current_step >= len(self.prices) - 1
        
        # Get next state
        state = self._get_observation()
        
        # Increment step
        self.current_step += 1
        
        return state, reward, done, {}
        
    def _get_observation(self) -> np.ndarray:
        """Get current state observation
        
        Returns:
            np.ndarray: State observation (window_size x n_features)
        """
        # TODO: Implement state observation logic
        return self.prices[self.current_step - self.window_size:self.current_step]
        
    def _calculate_reward(self) -> float:
        """Calculate reward for current step
        
        Returns:
            float: Reward value
        """
        # TODO: Implement reward calculation
        return 0.0
        
    def render(self, mode: str = 'human') -> Any:
        """Render environment state
        
        Args:
            mode (str): Rendering mode ('human' or 'rgb_array')
            
        Returns:
            Any: Rendering output depending on mode
        """
        # TODO: Implement rendering logic
        pass
