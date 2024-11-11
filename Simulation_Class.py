import matplotlib.pyplot as plt
import numpy as np
import pickle

from enum import Enum

class ActionPS(Enum):
    LEAVE = 0
    STAY = 1

class ActionPD(Enum):
    DEFECT = 0
    COOPERATE = 1

class State(Enum):
    PARTNER_DEFECTED = 0
    PARTNER_COOPERATED = 1

def get_state(action: ActionPD) -> State:
    return State.PARTNER_COOPERATED if action == ActionPD.COOPERATE else State.PARTNER_DEFECTED

class StrategyPS(Enum):
    ALWAYS_STAY = 0
    OUT_FOR_TAT = 1
    REVERSE_OUT_FOR_TAT = 2
    ALWAYS_LEAVE = 3
    RANDOM = 4

class StrategyPD(Enum):
    ALWAYS_COOPERATE = 0
    TFT = 1
    RTFT = 2
    ALWAYS_DEFECT = 3
    RANDOM = 4
    
strategy_names = {
    StrategyPS.ALWAYS_STAY : 'Always-Stay',
    StrategyPS.OUT_FOR_TAT : 'Out-For-Tat',
    StrategyPS.REVERSE_OUT_FOR_TAT : 'Reverse-OFT',
    StrategyPS.ALWAYS_LEAVE : 'Always-Leave',
    StrategyPS.RANDOM : 'Random (PS)',
    StrategyPD.ALWAYS_COOPERATE : 'Always-Cooperate',
    StrategyPD.TFT : 'Tit-For-Tat',
    StrategyPD.RTFT : 'Reverse-TFT',
    StrategyPD.ALWAYS_DEFECT : 'Always-Defect',
    StrategyPD.RANDOM : 'Random (PD)',
}

strategy_colors = {
    StrategyPS.ALWAYS_STAY : 'lightcoral',
    StrategyPS.OUT_FOR_TAT : 'lightsteelblue',
    StrategyPS.REVERSE_OUT_FOR_TAT : 'lightgreen',
    StrategyPS.ALWAYS_LEAVE : 'tan',
    StrategyPS.RANDOM : 'mediumpurple',
    StrategyPD.ALWAYS_COOPERATE : 'red',
    StrategyPD.TFT : 'blue',
    StrategyPD.RTFT : 'green',
    StrategyPD.ALWAYS_DEFECT : 'yellow',
    StrategyPD.RANDOM : 'purple',
}

# returns the probabilities of selection actions given the current state 
def boltzmann_exploration(q_table, state: State, temperature: float):
    exp = np.exp((q_table[state.value, :] - max(q_table[state.value, :])) / temperature)
    return exp / np.sum(exp)

def epsilon_greedy(q_table, state: State, epsilon: float) -> np.ndarray:
    if np.random.rand() < epsilon:
        return np.ones(len(q_table[state.value, :])) / len(q_table[state.value, :])
    else:
        prob = np.zeros(len(q_table[state.value, :]))
        prob[np.argmax(q_table[state.value, :])] = 1
        return prob
    
def greedy(q_table, state: State, _: float = None) -> np.ndarray:
    prob = np.zeros(len(q_table[state.value, :]))
    prob[np.argmax(q_table[state.value, :])] = 1
    return prob

# runs the Q-Learning algorithm on the provided qtable
# NOTE: alpha is the learning rate and gamma is the discount rate
def q_learning(qtable, next_qtable, state: State, action, 
               reward: float, new_state: State, alpha: float, gamma: float) -> None:
    qtable[state.value, action.value] = (1.0 - alpha) * qtable[state.value, action.value] + \
        alpha * (reward + gamma * np.max(next_qtable[new_state.value, :]))
    
def sarsa_learning(qtable, next_qtable, state: State, action, 
               reward: float, new_state: State, new_action, alpha: float, gamma: float) -> None:
    qtable[state.value, action.value] = (1.0 - alpha) * qtable[state.value, action.value] + \
        alpha * (reward + gamma * next_qtable[new_state.value, new_action.value])

class Agent:
    def __init__(self, learning_rate: float, temperature: float, discount_rate: float, 
                 delta_t: float, decay_type = None, policy = None,
                 last_action_pd: ActionPD = None, qtable_ps = None, qtable_pd = None):
        self.a = learning_rate
        self.t = temperature
        self.g = discount_rate
        self.delta_t = delta_t
        self.decay_type = "exponential" if decay_type is None else decay_type
        self.last_trajectory = None
        self.last_action_pd = np.random.choice([ActionPD.DEFECT, ActionPD.COOPERATE], 1) if last_action_pd == None else last_action_pd
        self.qtable_ps = np.zeros((2, 2)) if qtable_ps is None else qtable_ps
        self.qtable_pd = np.zeros((2, 2)) if qtable_pd is None else qtable_pd
        self.policy = boltzmann_exploration if policy is None else policy
        
        self.stay_streak = 0
        self.leave_streak = 0
        self.cooperate_streak = 0
        self.defect_streak = 0

        # Force agents to use Out For Tat Partner Selection Strategy
        # self.qtable_ps[0, 0] = 10
        # self.qtable_ps[1, 1] = 10
        
    # returns an action given the current state
    def get_action_ps(self, state: State, debug = False) -> ActionPS:
        temp = self.policy(self.qtable_ps, state, self.t)
        action = np.random.choice([ActionPS.LEAVE, ActionPS.STAY], p=temp)
        if debug:
            print("Action Probabilities: " + str(temp))
            print("Chosen Action: " + str(action))
        return action

        # Force Out for Tat
        # if state == State.PARTNER_COOPERATED:
        #     return ActionPS.STAY
        # else:
        #     return ActionPS.LEAVE

    def get_action_pd(self, state: State, debug = False) -> ActionPD:
        temp = self.policy(self.qtable_pd, state, self.t)
        action = np.random.choice([ActionPD.DEFECT, ActionPD.COOPERATE], p=temp)
        if debug:
            print("Action Probabilities: " + str(temp))
            print("Chosen Action: " + str(action))
        return action
    
    def update_reward(self, reward):
        pass
    
    # trains using trajectories from each round
    def train(self, trajectories, learning_mode, 
              last_trajectory = None, debug = False):
        
        # Author's Implementation

        # discounted_rewards = []
        # running_sum = 0
        # for trajectory in trajectories[::-1]:
        #     running_sum = trajectory[6] + self.g * running_sum
        #     discounted_rewards.append(running_sum)
        # discounted_rewards.reverse()

        # for idx, trajectory in enumerate(trajectories):
        #     # partner selection training
        #     q_learning(self.qtable_ps, self.qtable_pd, trajectory[0], trajectory[1], discounted_rewards[idx], trajectory[3], self.a, self.g)
        #     # prisoner's dilemma training
        #     q_learning(self.qtable_pd, self.qtable_ps, trajectory[3], trajectory[4], discounted_rewards[idx], trajectory[5], self.a, self.g)

        # Interpreted Implementation

        if learning_mode == "q_learning":
            for idx, trajectory in enumerate(trajectories):
                # partner selection training
                q_learning(self.qtable_ps, self.qtable_pd, trajectory[0], trajectory[1], trajectory[2], trajectory[3], self.a, self.g)
                
                # prisoner's dilemma training
                q_learning(self.qtable_pd, self.qtable_ps, trajectory[3], trajectory[4], trajectory[6], trajectory[5], self.a, self.g)
                
        elif learning_mode == "sarsa":
            for idx, trajectory in enumerate(trajectories):
                # partner selection training
                sarsa_learning(self.qtable_ps, self.qtable_pd, trajectory[0], trajectory[1], trajectory[2], trajectory[3], trajectory[4], self.a, self.g)
                if last_trajectory is not None:
                    # prisoner's dilemma training
                    sarsa_learning(self.qtable_pd, self.qtable_ps, last_trajectory[3], last_trajectory[4], last_trajectory[6], last_trajectory[5], trajectory[1], self.a, self.g)
                last_trajectory = trajectory
                
        # decrease temperature
        # self.t *= 0.01
        if self.decay_type == "exponential":
            self.t *= self.delta_t
        if self.decay_type == "linear":
            self.t -= self.delta_t
    
    def get_strategy_ps(self):
        if (self.qtable_ps[0, 0] < self.qtable_ps[0, 1] and self.qtable_ps[1, 0] < self.qtable_ps[1, 1]):
            return StrategyPS.ALWAYS_STAY
        elif (self.qtable_ps[0, 0] > self.qtable_ps[0, 1] and self.qtable_ps[1, 0] < self.qtable_ps[1, 1]):
            return StrategyPS.OUT_FOR_TAT
        elif (self.qtable_ps[0, 0] < self.qtable_ps[0, 1] and self.qtable_ps[1, 0] > self.qtable_ps[1, 1]):
            return StrategyPS.REVERSE_OUT_FOR_TAT
        elif (self.qtable_ps[0, 0] > self.qtable_ps[0, 1] and self.qtable_ps[1, 0] > self.qtable_ps[1, 1]):
            return StrategyPS.ALWAYS_LEAVE
        else:
            return StrategyPS.RANDOM

    def get_strategy_pd(self):
        if (self.qtable_pd[0, 0] < self.qtable_pd[0, 1] and self.qtable_pd[1, 0] < self.qtable_pd[1, 1]):
            return StrategyPD.ALWAYS_COOPERATE
        elif (self.qtable_pd[0, 0] > self.qtable_pd[0, 1] and self.qtable_pd[1, 0] < self.qtable_pd[1, 1]):
            return StrategyPD.TFT
        elif (self.qtable_pd[0, 0] < self.qtable_pd[0, 1] and self.qtable_pd[1, 0] > self.qtable_pd[1, 1]):
            return StrategyPD.RTFT
        elif (self.qtable_pd[0, 0] > self.qtable_pd[0, 1] and self.qtable_pd[1, 0] > self.qtable_pd[1, 1]):
            return StrategyPD.ALWAYS_DEFECT
        else:
            return StrategyPD.RANDOM
    
# returns the rewards of two agents in the prisoner's dilemma game
def prisoners_dilemma(a_i: ActionPD, a_j: ActionPD) -> tuple[float, float]:
    reward_table = np.array([[(1, 1), (5, 0)], [(0, 5), (3, 3)]])
    return reward_table[a_i.value, a_j.value]

class Simulation:
    def __init__(self, population: int, rounds: int, learning_rate: float, 
         starting_temperature: float, discount_rate: float, delta_t: float,
         decay_type: str, start_decay_episode: int, end_decay_episode: int,
         disposition: float, know_fresh_agent: float, prefer_same_pool: float, prefer_different_pool: float,
         learning_mode: str = "q_learning", policy_mode: str = "boltzmann"):
        self.population = population
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.temperature = starting_temperature
        self.discount_rate = discount_rate
        self.delta_t = delta_t
        self.decay_type = decay_type
        self.start_decay_episode = start_decay_episode
        self.end_decay_episode = end_decay_episode if end_decay_episode is not None else float("inf")
        self.disposition = disposition
        self.know_fresh_agent = know_fresh_agent
        self.prefer_same_pool = prefer_same_pool
        self.prefer_different_pool = prefer_different_pool
        self.learning_mode = learning_mode
        self.learn = q_learning if learning_mode == "q_learning" else sarsa_learning if learning_mode == "sarsa" else None
        self.policy_mode = policy_mode
        self.policy = greedy if policy_mode == "greedy" else epsilon_greedy if policy_mode == "epsilon_greedy" else boltzmann_exploration
        self.initialized = False

        if (population % 2 != 0):
            print("sdoo: population must be a multiple of two")
            return
        
        if prefer_same_pool + prefer_different_pool > 1.0:
            print("sdoo: prefer_same_pool + prefer_different_pool must be less than or equal to 1.0")
            return
        
        if learning_mode not in ["q_learning", "sarsa", "none"]:
            print("sdoo: learning_mode must be either 'q_learning' or 'sarsa' or 'none'")
            return
        
        if policy_mode not in ["boltzmann", "epsilon_greedy", "greedy"]:
            print("sdoo: policy_mode must be either 'boltzmann' or 'epsilon_greedy' or 'greedy'")
            return

        if decay_type not in ["exponential", "linear"]:
            print("sdoo: decay_type must be either 'exponential' or 'linear'")
        
    def initialize(self):
        if self.initialized:
            print("Simulation already initialized")
            return

        self.recorded_outcomes_pd = {
            (ActionPD.DEFECT, ActionPD.DEFECT): [],
            (ActionPD.DEFECT, ActionPD.COOPERATE): [],
            (ActionPD.COOPERATE, ActionPD.DEFECT): [],
            (ActionPD.COOPERATE, ActionPD.COOPERATE): [],
        }

        self.recorded_outcomes_ps = {
            (ActionPS.LEAVE, ActionPS.LEAVE): [],
            (ActionPS.LEAVE, ActionPS.STAY): [],
            (ActionPS.STAY, ActionPS.LEAVE): [],
            (ActionPS.STAY, ActionPS.STAY): [],
        }

        self.recorded_agent_strategy_pairings = {
            (StrategyPD.ALWAYS_COOPERATE, StrategyPD.ALWAYS_COOPERATE): [],
            (StrategyPD.ALWAYS_COOPERATE, StrategyPD.TFT): [],
            (StrategyPD.ALWAYS_COOPERATE, StrategyPD.RTFT): [],
            (StrategyPD.ALWAYS_COOPERATE, StrategyPD.ALWAYS_DEFECT): [],
            (StrategyPD.TFT, StrategyPD.TFT): [],
            (StrategyPD.TFT, StrategyPD.RTFT): [],
            (StrategyPD.TFT, StrategyPD.ALWAYS_DEFECT): [],
            (StrategyPD.RTFT, StrategyPD.RTFT): [],
            (StrategyPD.RTFT, StrategyPD.ALWAYS_DEFECT): [],
            (StrategyPD.ALWAYS_DEFECT, StrategyPD.ALWAYS_DEFECT): [],
        }

        self.recorded_outcome_changes = {
            ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [],
            ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.DEFECT)): [],
            ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.COOPERATE)): [],
            ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.DEFECT)): [],

            ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [],
            ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.DEFECT)): [],
            ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.COOPERATE)): [],
            ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.DEFECT)): [],

            ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [],
            ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.DEFECT)): [],
            ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.COOPERATE)): [],
            ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.DEFECT)): [],

            ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [],
            ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.DEFECT)): [],
            ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.COOPERATE)): [],
            ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.DEFECT)): [],
        }

        self.agent_qvales_ps = [[] for _ in range(4)]
        self.agent_qvales_pd = [[] for _ in range(4)]

        self.recorded_qvalues_ps = [[[] for _ in range(4)] for _ in range(self.population)]
        self.recorded_qvalues_pd = [[[] for _ in range(4)] for _ in range(self.population)]

        self.agent_pd_actions_per_episode = [[] for _ in range(4)]
        self.percentage_of_states_per_episode = [[] for _ in range(4)]
        
        self.agent_chosen_switches_per_episode = []
        self.agent_switches_per_episode = []

        self.t_values = []

        self.total_reward = []

        # Fix Randoms
        # np.random.seed(0)

        # Global Q-Table Test
        # qtable_ps = np.zeros((2, 2))
        # qtable_pd = np.zeros((2, 2))
        # agents = [Agent(learning_rate, temperature, discount_rate, qtable_ps=qtable_ps, qtable_pd=qtable_pd) for _ in range(population)]
        
        self.agents = [Agent(self.learning_rate, self.temperature, self.discount_rate, self.delta_t, "none", policy=self.policy) for _ in range(self.population)]
        self.unpaired = list(range(self.population))

        # Pair Agents
        self.pairs: tuple[int, int] = []
        while self.unpaired:
            i = self.unpaired.pop(np.random.randint(len(self.unpaired)))
            j = self.unpaired.pop(np.random.randint(len(self.unpaired)))
            self.pairs.append((i, j))

        self.probabilities_ps_defected = []
        self.probabilities_ps_cooperated = []
        self.probabilities_pd_defected = []
        self.probabilities_pd_cooperated = []
        self.max_probabilities_ps_defected = []
        self.max_probabilities_ps_cooperated = []
        self.max_probabilities_pd_defected = []
        self.max_probabilities_pd_cooperated = []
        self.new_probabilities_ps_defected = []
        self.new_probabilities_pd_defected = []
        self.new_probabilities_ps_cooperated = []
        self.new_probabilities_pd_cooperated = []
        self.new_max_probabilities_ps_defected = []
        self.new_max_probabilities_pd_defected = []
        self.new_max_probabilities_ps_cooperated = []
        self.new_max_probabilities_pd_cooperated = []
        self.strategies_ps = []
        self.strategies_pd = []
        self.new_strategies_ps = []
        self.new_strategies_pd = []
        self.stay_streaks = []
        self.leave_streaks = []
        self.cooperate_streaks = []
        self.defect_streaks = []
        self.avg_stay_streak = []
        self.avg_leave_streak = []
        self.avg_cooperate_streak = []
        self.avg_defect_streak = []

        self.initialized = True

    def run_episodes(self, episodes):
        if not self.initialized:
            print("Simulation not initialized")
            return

        for episode in range(episodes):
            # Add elements to lists where needed
            for (action1, action2) in self.recorded_outcomes_pd:
                self.recorded_outcomes_pd[(action1, action2)].append(0)
            for (action1, action2) in self.recorded_outcomes_ps:
                self.recorded_outcomes_ps[(action1, action2)].append(0)
            for (strategy1, strategy2) in self.recorded_agent_strategy_pairings:
                self.recorded_agent_strategy_pairings[(strategy1, strategy2)].append(0)
            for (outcome1, outcome2) in self.recorded_outcome_changes:
                self.recorded_outcome_changes[(outcome1, outcome2)].append(0)
            for idx in range(len(self.agent_qvales_ps)):
                self.agent_qvales_ps[idx].append(0)
            for idx in range(len(self.agent_qvales_pd)):
                self.agent_qvales_pd[idx].append(0)
            for idx in range(len(self.recorded_qvalues_ps)):
                for idx2 in range(len(self.recorded_qvalues_ps[idx])):
                    self.recorded_qvalues_ps[idx][idx2].append(0)
            for idx in range(len(self.recorded_qvalues_pd)):
                for idx2 in range(len(self.recorded_qvalues_pd[idx])):
                    self.recorded_qvalues_pd[idx][idx2].append(0)
            for idx in range(len(self.agent_pd_actions_per_episode)):
                self.agent_pd_actions_per_episode[idx].append(0)
            for idx in range(len(self.percentage_of_states_per_episode)):
                self.percentage_of_states_per_episode[idx].append(0)
            self.agent_chosen_switches_per_episode.append(0)
            self.agent_switches_per_episode.append(0)
            self.total_reward.append(0)

            # Record agent Q-Values
            for agent_idx in range(len(self.recorded_qvalues_ps)):
                for idx in range(len(self.recorded_qvalues_ps[agent_idx])):
                    self.recorded_qvalues_ps[agent_idx][idx][episode] = self.agents[agent_idx].qtable_ps.ravel()[idx]

            for agent_idx in range(len(self.recorded_qvalues_pd)):
                for idx in range(len(self.recorded_qvalues_pd[agent_idx])):
                    self.recorded_qvalues_pd[agent_idx][idx][episode] = self.agents[agent_idx].qtable_pd.ravel()[idx]

            trajectories = [[] for _ in range(self.population)]

            if episode < self.start_decay_episode:
                for agent in self.agents:
                    agent.decay_type = "none"
            if self.start_decay_episode <= episode < self.end_decay_episode:
                for agent in self.agents:
                    agent.decay_type = self.decay_type
            if episode >= self.end_decay_episode:
                for agent in self.agents:
                    agent.decay_type = "none"
            for round in range(self.rounds):

                # Partner Selection
                temp_pairs = []
                switch_leave_pool = []
                switch_stay_pool = []
                stay_pool = []

                for (i, j) in self.pairs:
                    s_i = get_state(self.agents[j].last_action_pd)
                    s_j = get_state(self.agents[i].last_action_pd)
                    self.percentage_of_states_per_episode[s_i.value][episode] += 1
                    self.percentage_of_states_per_episode[s_j.value][episode] += 1

                    a_i = self.agents[i].get_action_ps(s_i)
                    a_j = self.agents[j].get_action_ps(s_j)

                    if a_i == ActionPS.STAY:
                        self.agents[i].stay_streak += 1
                        self.agents[i].leave_streak = 0
                    else:
                        self.agents[i].stay_streak = 0
                        self.agents[i].leave_streak += 1
                    if a_j == ActionPS.STAY:
                        self.agents[j].stay_streak += 1
                        self.agents[j].leave_streak = 0
                    else:
                        self.agents[j].stay_streak = 0
                        self.agents[j].leave_streak += 1

                    prob_ps_defected_i = self.policy(self.agents[i].qtable_ps, State.PARTNER_DEFECTED, self.agents[i].t)
                    prob_ps_cooperated_i = self.policy(self.agents[i].qtable_ps, State.PARTNER_COOPERATED, self.agents[i].t)
                    prob_ps_defected_j = self.policy(self.agents[j].qtable_ps, State.PARTNER_DEFECTED, self.agents[j].t)
                    prob_ps_cooperated_j = self.policy(self.agents[j].qtable_ps, State.PARTNER_COOPERATED, self.agents[j].t)

                    self.probabilities_ps_defected.append(prob_ps_defected_i[0])
                    self.probabilities_ps_defected.append(prob_ps_defected_j[0])
                    self.probabilities_ps_cooperated.append(prob_ps_cooperated_i[0])
                    self.probabilities_ps_cooperated.append(prob_ps_cooperated_j[0])
                    self.max_probabilities_ps_defected.append(max(prob_ps_defected_i))
                    self.max_probabilities_ps_defected.append(max(prob_ps_defected_j))
                    self.max_probabilities_ps_cooperated.append(max(prob_ps_cooperated_i))
                    self.max_probabilities_ps_cooperated.append(max(prob_ps_cooperated_j))

                    if a_i == ActionPS.LEAVE or a_j == ActionPS.LEAVE:
                        if a_i == ActionPS.LEAVE:
                            switch_leave_pool.append(i)
                            self.agents[i].last_action_ps = ActionPS.LEAVE
                            self.agents[i].last_result_ps = "split"
                        else:
                            switch_stay_pool.append(i)
                            self.agents[i].last_action_ps = ActionPS.STAY
                            self.agents[i].last_result_ps = "split"
                        if a_j == ActionPS.LEAVE:
                            switch_leave_pool.append(j)
                            self.agents[j].last_action_ps = ActionPS.LEAVE
                            self.agents[j].last_result_ps = "split"
                        else:
                            switch_stay_pool.append(j)
                            self.agents[j].last_action_ps = ActionPS.STAY
                            self.agents[j].last_result_ps = "split"
                        self.agent_switches_per_episode[episode] += 2
                        self.agent_chosen_switches_per_episode[episode] += int(a_i == ActionPS.LEAVE) + int(a_j == ActionPS.LEAVE)
                    else:
                        stay_pool.append((i, j))
                        self.agents[i].last_action_ps = ActionPS.STAY
                        self.agents[i].last_result_ps = "stay"
                        self.agents[j].last_action_ps = ActionPS.STAY
                        self.agents[j].last_result_ps = "stay"

                    # Partner Selection Rewards Test
                    # r_i = 1 if a_j == ActionPS.STAY else -1
                    # r_j = 1 if a_i == ActionPS.STAY else -1
                    r_i = 0
                    r_j = 0

                    trajectories[i].append((s_i, a_i, r_i))
                    trajectories[j].append((s_j, a_j, r_j))
                    
                # Pair Agents in switch_pool and switched_pool
                while switch_leave_pool and switch_stay_pool:
                    full_pool = switch_leave_pool + switch_stay_pool
                    i = full_pool.pop(np.random.randint(len(full_pool)))
                    i_pool = switch_leave_pool if i in switch_leave_pool else switch_stay_pool
                    i_pool.remove(i)
                    other_pool = switch_leave_pool if i_pool == switch_stay_pool else switch_stay_pool
                    rand_choice = np.random.rand()
                    if (rand_choice < self.prefer_same_pool) and i_pool:
                        j_pool = i_pool
                        j = j_pool.pop(np.random.randint(len(j_pool)))
                    elif (rand_choice < self.prefer_same_pool + self.prefer_different_pool) and other_pool:
                        j_pool = other_pool
                        j = j_pool.pop(np.random.randint(len(j_pool)))
                    else:
                        j = full_pool.pop(np.random.randint(len(full_pool)))
                        j_pool = switch_leave_pool if j in switch_leave_pool else switch_stay_pool
                        j_pool.remove(j)
                    temp_pairs.append((i, j))

                # Pair Agents in switch_pool with each other
                while len(switch_leave_pool) > 1:
                    i = switch_leave_pool.pop(np.random.randint(len(switch_leave_pool)))
                    j = switch_leave_pool.pop(np.random.randint(len(switch_leave_pool)))
                    temp_pairs.append((i, j))

                # Pair Agents in switched_pool with each other
                while len(switch_stay_pool) > 1:
                    i = switch_stay_pool.pop(np.random.randint(len(switch_stay_pool)))
                    j = switch_stay_pool.pop(np.random.randint(len(switch_stay_pool)))
                    temp_pairs.append((i, j))
                
                # Combine new pairs with stay_pool pairs
                pairs = temp_pairs + stay_pool

                # Prisoner's Dilemma
                for (i, j) in pairs:
                    strategy_i = self.agents[i].get_strategy_pd()
                    strategy_j = self.agents[j].get_strategy_pd()
                    if (strategy_i, strategy_j) in self.recorded_agent_strategy_pairings:
                        self.recorded_agent_strategy_pairings[(strategy_i, strategy_j)][episode] += 1
                    elif (strategy_j, strategy_i) in self.recorded_agent_strategy_pairings:
                        self.recorded_agent_strategy_pairings[(strategy_j, strategy_i)][episode] += 1
                    
                    # If agent split, use random action
                    if (self.agents[i].last_result_ps == "split" and self.agents[i].last_action_ps == "split") and np.random.rand() > self.know_fresh_agent:
                        s_i = State.PARTNER_COOPERATED if np.random.rand() < self.disposition else State.PARTNER_DEFECTED
                        s_j = State.PARTNER_COOPERATED if np.random.rand() < self.disposition else State.PARTNER_DEFECTED
                    else:
                        s_i = get_state(self.agents[j].last_action_pd)
                        s_j = get_state(self.agents[i].last_action_pd)
                    self.percentage_of_states_per_episode[s_i.value + 2][episode] += 1
                    self.percentage_of_states_per_episode[s_j.value + 2][episode] += 1

                    a_i = self.agents[i].get_action_pd(s_i)
                    a_j = self.agents[j].get_action_pd(s_j)

                    if a_i == ActionPD.COOPERATE:
                        self.agents[i].cooperate_streak += 1
                        self.agents[i].defect_streak = 0
                    else:
                        self.agents[i].cooperate_streak = 0
                        self.agents[i].defect_streak += 1
                    if a_j == ActionPD.COOPERATE:
                        self.agents[j].cooperate_streak += 1
                        self.agents[j].defect_streak = 0
                    else:
                        self.agents[j].cooperate_streak = 0
                        self.agents[j].defect_streak += 1

                    prob_pd_defected_i = self.policy(self.agents[i].qtable_pd, State.PARTNER_DEFECTED, self.agents[i].t)
                    prob_pd_cooperated_i = self.policy(self.agents[i].qtable_pd, State.PARTNER_COOPERATED, self.agents[i].t)
                    prob_pd_defected_j = self.policy(self.agents[j].qtable_pd, State.PARTNER_DEFECTED, self.agents[j].t)
                    prob_pd_cooperated_j = self.policy(self.agents[j].qtable_pd, State.PARTNER_COOPERATED, self.agents[j].t)

                    self.probabilities_pd_defected.append(prob_pd_defected_i[0])
                    self.probabilities_pd_defected.append(prob_pd_defected_j[0])
                    self.probabilities_pd_cooperated.append(prob_pd_cooperated_i[0])
                    self.probabilities_pd_cooperated.append(prob_pd_cooperated_j[0])
                    self.max_probabilities_pd_defected.append(max(prob_pd_defected_i))
                    self.max_probabilities_pd_defected.append(max(prob_pd_defected_j))
                    self.max_probabilities_pd_cooperated.append(max(prob_pd_cooperated_i))
                    self.max_probabilities_pd_cooperated.append(max(prob_pd_cooperated_j))
                    self.stay_streaks.append(self.agents[i].stay_streak)
                    self.stay_streaks.append(self.agents[j].stay_streak)
                    self.leave_streaks.append(self.agents[i].leave_streak)
                    self.leave_streaks.append(self.agents[j].leave_streak)
                    self.cooperate_streaks.append(self.agents[i].cooperate_streak)
                    self.cooperate_streaks.append(self.agents[j].cooperate_streak)
                    self.defect_streaks.append(self.agents[i].defect_streak)
                    self.defect_streaks.append(self.agents[j].defect_streak)

                    r_i, r_j = prisoners_dilemma(a_i, a_j)
                    self.total_reward[episode] += r_i + r_j

                    ns_i = get_state(a_j)
                    ns_j = get_state(a_i)
                    self.recorded_outcomes_pd[(a_i, a_j)][episode] += 1
                    self.agents[i].last_action_pd = a_i
                    self.agents[j].last_action_pd = a_j

                    # Record Trajectories
                    t = trajectories[i][round]
                    trajectories[i][round] = (t[0], t[1], t[2], s_i, a_i, ns_i, r_i)
                    t = trajectories[j][round]
                    trajectories[j][round] = (t[0], t[1], t[2], s_j, a_j, ns_j, r_j)

                    # Record Actions taken
                    self.agent_pd_actions_per_episode[2 * (s_i.value - 2) + a_i.value][episode] += 1
                    self.agent_pd_actions_per_episode[2 * (s_j.value - 2) + a_j.value][episode] += 1

            for idx, agent in enumerate(self.agents):
                agent.train(trajectories[idx], self.learning_mode, last_trajectory=agent.last_trajectory)
            for idx, agent in enumerate(self.agents):
                agent.last_trajectory = trajectories[idx][-1]
            temps = [agent.t for agent in self.agents]
            self.t_values.append(np.mean(temps))
            for idx in range(len(self.agent_qvales_ps)):
                self.agent_qvales_ps[idx][episode] = np.sum([agent.qtable_ps.ravel()[idx] for agent in self.agents]) / self.population

            for idx in range(len(self.agent_qvales_pd)):
                self.agent_qvales_pd[idx][episode] = np.sum([agent.qtable_pd.ravel()[idx] for agent in self.agents]) / self.population
            
            self.recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.DEFECT)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.COOPERATE)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.DEFECT)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.COOPERATE)][episode] /= (self.rounds * self.population / 2)

            self.recorded_outcomes_ps[(ActionPS.LEAVE, ActionPS.LEAVE)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_ps[(ActionPS.LEAVE, ActionPS.STAY)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_ps[(ActionPS.STAY, ActionPS.LEAVE)][episode] /= (self.rounds * self.population / 2)
            self.recorded_outcomes_ps[(ActionPS.STAY, ActionPS.STAY)][episode] /= (self.rounds * self.population / 2)

            self.agent_pd_actions_per_episode[0][episode] /= (self.rounds * self.population)
            self.agent_pd_actions_per_episode[1][episode] /= (self.rounds * self.population)
            self.agent_pd_actions_per_episode[2][episode] /= (self.rounds * self.population)
            self.agent_pd_actions_per_episode[3][episode] /= (self.rounds * self.population)

            self.percentage_of_states_per_episode[0][episode] /= (self.rounds * self.population)
            self.percentage_of_states_per_episode[1][episode] /= (self.rounds * self.population)
            self.percentage_of_states_per_episode[2][episode] /= (self.rounds * self.population)
            self.percentage_of_states_per_episode[3][episode] /= (self.rounds * self.population)

            for idx, agent in enumerate(self.agents):
                self.strategies_ps.append(agent.get_strategy_ps())
                self.strategies_pd.append(agent.get_strategy_pd())

            for agent_trajectories in trajectories:
                for idx in range(self.rounds - 1):
                    outcome = (agent_trajectories[idx][4], ActionPD.COOPERATE if agent_trajectories[idx][5] == State.PARTNER_COOPERATED else ActionPD.DEFECT)
                    next_outcome = (agent_trajectories[idx + 1][4], ActionPD.COOPERATE if agent_trajectories[idx + 1][5] == State.PARTNER_COOPERATED else ActionPD.DEFECT)

                    self.recorded_outcome_changes[(outcome, next_outcome)][episode] += 1
            
            mean_probabilities_ps_defected = np.mean(self.probabilities_ps_defected)
            mean_probabilities_ps_cooperated = np.mean(self.probabilities_ps_cooperated)
            mean_probabilities_pd_defected = np.mean(self.probabilities_pd_defected)
            mean_probabilities_pd_cooperated = np.mean(self.probabilities_pd_cooperated)
            mean_max_probabilities_ps_defected = np.mean(self.max_probabilities_ps_defected)
            mean_max_probabilities_ps_cooperated = np.mean(self.max_probabilities_ps_cooperated)
            mean_max_probabilities_pd_defected = np.mean(self.max_probabilities_pd_defected)
            mean_max_probabilities_pd_cooperated = np.mean(self.max_probabilities_pd_cooperated)

            self.new_probabilities_ps_defected.append(mean_probabilities_ps_defected)
            self.new_probabilities_ps_cooperated.append(mean_probabilities_ps_cooperated)
            self.new_probabilities_pd_defected.append(mean_probabilities_pd_defected)
            self.new_probabilities_pd_cooperated.append(mean_probabilities_pd_cooperated)
            self.new_max_probabilities_ps_defected.append(mean_max_probabilities_ps_defected)
            self.new_max_probabilities_ps_cooperated.append(mean_max_probabilities_ps_cooperated)
            self.new_max_probabilities_pd_defected.append(mean_max_probabilities_pd_defected)
            self.new_max_probabilities_pd_cooperated.append(mean_max_probabilities_pd_cooperated)

            self.avg_stay_streak.append(np.mean(self.stay_streaks))
            self.avg_leave_streak.append(np.mean(self.leave_streaks))
            self.avg_cooperate_streak.append(np.mean(self.cooperate_streaks))
            self.avg_defect_streak.append(np.mean(self.defect_streaks))

            self.new_strategies_ps.append(self.strategies_ps)
            self.new_strategies_pd.append(self.strategies_pd)

            self.probabilities_ps_defected = []
            self.probabilities_ps_cooperated = []
            self.probabilities_pd_defected = []
            self.probabilities_pd_cooperated = []
            self.max_probabilities_ps_defected = []
            self.max_probabilities_ps_cooperated = []
            self.max_probabilities_pd_defected = []
            self.max_probabilities_pd_cooperated = []
            self.strategies_ps = []
            self.strategies_pd = []
            self.stay_streaks = []
            self.leave_streaks = []
            self.cooperate_streaks = []
            self.defect_streaks = []
    
        self.strat_always_leave = [sum([1 for strategy in strategies if strategy == StrategyPS.ALWAYS_LEAVE]) for strategies in self.new_strategies_ps]
        self.strat_out_for_tat = [sum([1 for strategy in strategies if strategy == StrategyPS.OUT_FOR_TAT]) for strategies in self.new_strategies_ps]
        self.strat_reverse_out_for_tat = [sum([1 for strategy in strategies if strategy == StrategyPS.REVERSE_OUT_FOR_TAT]) for strategies in self.new_strategies_ps]
        self.strat_always_stay = [sum([1 for strategy in strategies if strategy == StrategyPS.ALWAYS_STAY]) for strategies in self.new_strategies_ps]
        self.strat_always_defect = [sum([1 for strategy in strategies if strategy == StrategyPD.ALWAYS_DEFECT]) for strategies in self.new_strategies_pd]
        self.strat_tft = [sum([1 for strategy in strategies if strategy == StrategyPD.TFT]) for strategies in self.new_strategies_pd]
        self.strat_rtft = [sum([1 for strategy in strategies if strategy == StrategyPD.RTFT]) for strategies in self.new_strategies_pd]
        self.strat_always_cooperate = [sum([1 for strategy in strategies if strategy == StrategyPD.ALWAYS_COOPERATE]) for strategies in self.new_strategies_pd]

        # for agent_idx in range(recorded_qvalues_ps):
        #         for idx in range(recorded_qvalues_ps[agent_idx]):
        #             recorded_qvalues_ps[agent_idx][idx][episodes] = agent.qtable_ps.ravel()[idx]

        num_strategies_ps = [0 for _ in StrategyPS]
        num_strategies_pd = [0 for _ in StrategyPD]

        strategy_combinations = np.zeros((len(StrategyPS), len(StrategyPD)))

        # Determine Agent Strategies
        for idx, agent in enumerate(self.agents):
            strategy_ps = agent.get_strategy_ps()
            strategy_pd = agent.get_strategy_pd()
            
            num_strategies_ps[strategy_ps.value] += 1
            num_strategies_pd[strategy_pd.value] += 1
            strategy_combinations[strategy_ps.value, strategy_pd.value] += 1

        results = {
            "recorded_outcomes_pd": self.recorded_outcomes_pd,
            "recorded_outcomes_ps": self.recorded_outcomes_ps,
            "recorded_agent_strategy_pairings": self.recorded_agent_strategy_pairings,
            "recorded_outcome_changes": self.recorded_outcome_changes,
            "agent_qvales_ps": self.agent_qvales_ps,
            "agent_qvales_pd": self.agent_qvales_pd,
            "recorded_qvalues_ps": self.recorded_qvalues_ps,
            "recorded_qvalues_pd": self.recorded_qvalues_pd,
            "agent_pd_actions_per_episode": self.agent_pd_actions_per_episode,
            "percentage_of_states_per_episode": self.percentage_of_states_per_episode,
            "agent_chosen_switches_per_episode": self.agent_chosen_switches_per_episode,
            "agent_switches_per_episode": self.agent_switches_per_episode,
            "total_reward": self.total_reward,
            "ps_strategies": {
                StrategyPS.ALWAYS_LEAVE: self.strat_always_leave,
                StrategyPS.OUT_FOR_TAT: self.strat_out_for_tat,
                StrategyPS.REVERSE_OUT_FOR_TAT: self.strat_reverse_out_for_tat,
                StrategyPS.ALWAYS_STAY: self.strat_always_stay,
            },
            "pd_strategies": {
                StrategyPD.ALWAYS_DEFECT: self.strat_always_defect,
                StrategyPD.TFT: self.strat_tft,
                StrategyPD.RTFT: self.strat_rtft,
                StrategyPD.ALWAYS_COOPERATE: self.strat_always_cooperate,
            },
            "new_probabilities_ps_defected": self.new_probabilities_ps_defected,
            "new_probabilities_ps_cooperated": self.new_probabilities_ps_cooperated,
            "new_probabilities_pd_defected": self.new_probabilities_pd_defected,
            "new_probabilities_pd_cooperated": self.new_probabilities_pd_cooperated,
            "new_max_probabilities_ps_defected": self.new_max_probabilities_ps_defected,
            "new_max_probabilities_ps_cooperated": self.new_max_probabilities_ps_cooperated,
            "new_max_probabilities_pd_defected": self.new_max_probabilities_pd_defected,
            "new_max_probabilities_pd_cooperated": self.new_max_probabilities_pd_cooperated,
            "population": self.population,
            "rounds": self.rounds,
            "num_strategies_ps": num_strategies_ps,
            "num_strategies_pd": num_strategies_pd,
            "strategy_combinations": strategy_combinations,
            "temps": self.t_values,
            "avg_stay_streak": self.avg_stay_streak,
            "avg_leave_streak": self.avg_leave_streak,
            "avg_cooperate_streak": self.avg_cooperate_streak,
            "avg_defect_streak": self.avg_defect_streak,
        }
        return results

    def set_learning(self, learning_mode: str):
        if learning_mode not in ["q_learning", "sarsa", "none"]:
            print("Invalid learning mode")
            return
        self.learning_mode = learning_mode

    def set_policy(self, policy_mode: str):
        if policy_mode not in ["boltzmann", "epsilon_greedy", "greedy"]:
            print("Invalid policy mode")
            return
        self.policy_mode = policy_mode
        self.policy = greedy if policy_mode == "greedy" else epsilon_greedy if policy_mode == "epsilon_greedy" else boltzmann_exploration
        for agent in self.agents:
            agent.policy = self.policy

def average_results(results):
    keys = results[0].keys()
    averaged_results = {}
    for key in keys:
        if isinstance(results[0][key], dict):
            averaged_results[key] = {}
            for subkey in results[0][key].keys():
                try:
                    averaged_results[key][subkey] = np.mean([result[key][subkey] for result in results], axis=0)
                except TypeError as e:
                    averaged_results[key][subkey] = [result[key][subkey] for result in results]
                except ValueError as ve:
                    print(key)
                    raise ve
        else:
            try:
                averaged_results[key] = np.mean([result[key] for result in results], axis=0)
            except TypeError as e:
                averaged_results[key] = [result[key] for result in results]
            except ValueError as ve:
                print(key)
                raise ve
    return averaged_results

def main():

    ##############################################################################################################
    # Simulation Parameters
    ##############################################################################################################

    episodes = 5000
    reps = 5

    base_params = {"population": 20,   # Agent Population Size (Must be a multiple of 2)
        "rounds": 20,            # Rounds per Episode
        "learning_rate": 0.05,   # Alpha (Learning Rate)
        "learning_mode": "q_learning", # Learning Mode (q_learning or sarsa)
        "policy_mode": "epsilon_greedy", # Policy Mode (epsilon_greedy or boltzmann)
        "starting_temperature": 85,       # Starting Boltzmann Temperature 
        "discount_rate": 0.992,   # Gamma (Discount Rate)
        "delta_t": 0.99,         # Boltzmann Temperature Decay Rate
        "decay_type": "exponential", # Decay Type (exponential or linear)
        "start_decay_episode": 0, # Episode to Start Decay
        "end_decay_episode": None, # Episode to End Decay
        "disposition": 0.0,      # Disposition to Assume Cooperation
        "know_fresh_agent": 1.0, # Probability of Knowing Fresh Agent's Previous Action
        "prefer_same_pool": 0.0, # Probability of Choosing Same Pool Partner
        "prefer_different_pool": 0.0, # Probability of Choosing Different Pool Partner
        }

    # Can be empty to run just the base parameters
    tests = { 
    }

    ##############################################################################################################
    # Run Simulation
    ##############################################################################################################

    if not tests:
        tests = {"learning_mode": ["q_learning"]}

    params = base_params.copy()
    all_results_reps = {}
    all_results_average = {}

    # timestamp for filename
    import time
    import datetime
    ts = time.time()
    stts = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')

    for param_name, param_values in tests.items():
        param_results_reps = {}
        param_results_avg = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            params = base_params.copy()
            params[param_name] = param_value
            output = []
            for _ in range(reps):
                base = Simulation(**params)
                base.initialize()
                output.append(base.run_episodes(episodes))
            param_results_reps[param_value] = {"params": params.copy() | {"reps": reps}, "output": output}
            output_avg = average_results(output)
            param_results_avg[param_value] = {"params": params.copy() | {"reps": reps}, "output": output_avg}
        with open(f'reps/reps_{stts}.pkl', 'wb') as f:
            pickle.dump(param_results_reps, f)
        with open(f'avg_{stts}.pkl', 'wb') as f:
            pickle.dump(param_results_avg, f)
        all_results_reps[param_name] = param_results_reps
        all_results_average[param_name] = param_results_avg

if __name__ == "__main__":
    main()