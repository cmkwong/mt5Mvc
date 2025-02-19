import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Tuple, Optional

class StockAgent:
    """Template for stock trading agent
    
    Attributes:
        env: Trading environment instance
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        memory (deque): Replay memory buffer
        batch_size (int): Training batch size
        gamma (float): Discount factor
        epsilon (float): Exploration rate
        epsilon_min (float): Minimum exploration rate
        epsilon_decay (float): Exploration rate decay
        learning_rate (float): Learning rate
        model: Neural network model
        target_model: Target network model
        optimizer: Model optimizer
    """
    
    def __init__(self, env, state_dim: int, action_dim: int, 
                 memory_size: int = 10000, batch_size: int = 64,
                 gamma: float = 0.95, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001):
        """Initialize agent
        
        Args:
            env: Trading environment instance
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            memory_size (int): Size of replay memory
            batch_size (int): Training batch size
            gamma (float): Discount factor
            epsilon (float): Exploration rate
            epsilon_min (float): Minimum exploration rate
            epsilon_decay (float): Exploration rate decay
            learning_rate (float): Learning rate
        """
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        
        # Initialize model and target model
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.learning_rate)
        
    def _build_model(self) -> nn.Module:
        """Build neural network model
        
        Returns:
            nn.Module: Neural network model
        """
        # TODO: Implement model architecture
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
    def update_target_model(self):
        """Update target model weights"""
        self.target_model.load_state_dict(self.model.state_dict())
        
    def remember(self, state: np.ndarray, action: int, 
                reward: float, next_state: np.ndarray, done: bool):
        """Store experience in memory
        
        Args:
            state (np.ndarray): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (np.ndarray): Next state
            done (bool): Whether episode is done
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: np.ndarray) -> int:
        """Select action using epsilon-greedy policy
        
        Args:
            state (np.ndarray): Current state
            
        Returns:
            int: Selected action
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values).item()
        
    def replay(self) -> Optional[float]:
        """Train on batch from memory
        
        Returns:
            Optional[float]: Loss value if training occurred, None otherwise
        """
        if len(self.memory) < self.batch_size:
            return None
            
        minibatch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.LongTensor(np.array([i[1] for i in minibatch]))
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch]))
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch]))
        
        # Calculate target Q values
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).detach().max(1)[0]
        target = rewards + (1 - dones) * self.gamma * next_q
        
        # Calculate loss and update model
        loss = nn.MSELoss()(current_q.squeeze(), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def load(self, filename: str):
        """Load model weights from file
        
        Args:
            filename (str): Path to model weights file
        """
        self.model.load_state_dict(torch.load(filename))
        self.update_target_model()
        
    def save(self, filename: str):
        """Save model weights to file
        
        Args:
            filename (str): Path to save model weights
        """
        torch.save(self.model.state_dict(), filename)
