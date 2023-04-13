import collections
import numpy as np
from collections import namedtuple, deque

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])

class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def _group_list(self, items, lens):
        """
        Unflat the list of items by lens
        :param items: list of items
        :param lens: list of integers
        :return: list of list of items grouped by lengths
        """
        res = []
        cur_ofs = 0
        for g_len in lens:
            res.append(items[cur_ofs:cur_ofs + g_len])
            cur_ofs += g_len
        return res

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # The vectorized Gym API allows a single client-side environment to control a vector of remotes.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [-1] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = np.random.randint(self.pool[0].get_action_space_size())  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent.getActionIndex(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = self._group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n = env.step(action_n)
                else:
                    next_state, r, is_done = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res

class ExperienceSourceCMK:
    def __init__(self, env, agent, steps_count):
        self.env = env
        self.agent = agent
        self.steps_count = steps_count
        self.doneRewards, self.doneSteps = None, None

    def init_source(self):
        history = []
        cur_reward, cur_step = 0.0, 0.0
        state = self.env.reset()
        return state, history, cur_reward, cur_step

    def __iter__(self):
        state, history, cur_reward, cur_step = self.init_source()
        while True:
            if state is None:
                action = np.random.randint(self.env.get_action_space_size())
            else:
                action, _ = self.agent.getActionIndex([state])
            next_obs, r, is_done = self.env.step(action[0])
            cur_reward += r
            cur_step += 1
            history.append(Experience(state=state, action=action, reward=r, done=is_done))
            state = next_obs
            if len(history) == self.steps_count:
                yield history
                history = []
            if is_done:
                self.doneRewards = cur_reward
                self.doneSteps = cur_step
                state, history, cur_reward, cur_step = self.init_source()

    def pop_rewards_steps(self):
        if self.doneRewards != None and self.doneSteps != None:
            doneRewards_doneSteps = [self.doneRewards, self.doneSteps]
            self.doneRewards, self.doneSteps = None, None
            return doneRewards_doneSteps
        return False

# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))
class ExperienceSourceFirstLast(ExperienceSourceCMK):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__(): # step = 1: 2 experiences
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)