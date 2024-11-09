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
    def __init__(self, population: int, rounds: int, learning_rate: float, temperature: float, discount_rate: float, delta_t: float,
         decay_type: str,
         disposition: float, know_fresh_agent: float, prefer_same_pool: float, prefer_different_pool: float,
         learning_mode: str = "q_learning", policy_mode: str = "boltzmann"):
        self.population = population
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.discount_rate = discount_rate
        self.delta_t = delta_t
        self.decay_type = decay_type
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
        
        self.agents = [Agent(self.learning_rate, self.temperature, self.discount_rate, self.delta_t, self.decay_type, policy=self.policy) for _ in range(self.population)]
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

def sdoo(population: int, rounds: int, episodes: int, learning_rate: float, temperature: float, discount_rate: float, delta_t: float,
         disposition: float, know_fresh_agent: float, prefer_same_pool: float, prefer_different_pool: float, learning_mode: str = "q_learning",
         do_plot: bool = True):
    if (population % 2 != 0):
        print("sdoo: population must be a multiple of two")
        return
    if prefer_same_pool + prefer_different_pool > 1.0:
        print("sdoo: prefer_same_pool + prefer_different_pool must be less than or equal to 1.0")
        return
    
    recorded_outcomes_pd = {
        (ActionPD.DEFECT, ActionPD.DEFECT): [0 for _ in range(episodes)],
        (ActionPD.DEFECT, ActionPD.COOPERATE): [0 for _ in range(episodes)],
        (ActionPD.COOPERATE, ActionPD.DEFECT): [0 for _ in range(episodes)],
        (ActionPD.COOPERATE, ActionPD.COOPERATE): [0 for _ in range(episodes)],
    }

    recorded_agent_strategy_pairings = {
        # (a, b): [0 for _ in range(episodes)] for a in StrategyPD for b in StrategyPD
        (StrategyPD.ALWAYS_COOPERATE, StrategyPD.ALWAYS_COOPERATE): [0 for _ in range(episodes)],
        (StrategyPD.ALWAYS_COOPERATE, StrategyPD.TFT): [0 for _ in range(episodes)],
        (StrategyPD.ALWAYS_COOPERATE, StrategyPD.RTFT): [0 for _ in range(episodes)],
        (StrategyPD.ALWAYS_COOPERATE, StrategyPD.ALWAYS_DEFECT): [0 for _ in range(episodes)],
        # (StrategyPD.ALWAYS_COOPERATE, StrategyPD.RANDOM): [0 for _ in range(episodes)],
        (StrategyPD.TFT, StrategyPD.TFT): [0 for _ in range(episodes)],
        (StrategyPD.TFT, StrategyPD.RTFT): [0 for _ in range(episodes)],
        (StrategyPD.TFT, StrategyPD.ALWAYS_DEFECT): [0 for _ in range(episodes)],
        # (StrategyPD.TFT, StrategyPD.RANDOM): [0 for _ in range(episodes)],
        (StrategyPD.RTFT, StrategyPD.RTFT): [0 for _ in range(episodes)],
        (StrategyPD.RTFT, StrategyPD.ALWAYS_DEFECT): [0 for _ in range(episodes)],
        # (StrategyPD.RTFT, StrategyPD.RANDOM): [0 for _ in range(episodes)],
        (StrategyPD.ALWAYS_DEFECT, StrategyPD.ALWAYS_DEFECT): [0 for _ in range(episodes)],
        # (StrategyPD.ALWAYS_DEFECT, StrategyPD.RANDOM): [0 for _ in range(episodes)],
        # (StrategyPD.RANDOM, StrategyPD.RANDOM): [0 for _ in range(episodes)],
    }

    recorded_outcome_changes = {
        ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.DEFECT)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.DEFECT)): [0 for _ in range(episodes)],

        ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.DEFECT)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.COOPERATE, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.DEFECT)): [0 for _ in range(episodes)],

        ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.COOPERATE, ActionPD.DEFECT)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.COOPERATE), (ActionPD.DEFECT, ActionPD.DEFECT)): [0 for _ in range(episodes)],

        ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.COOPERATE, ActionPD.DEFECT)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.COOPERATE)): [0 for _ in range(episodes)],
        ((ActionPD.DEFECT, ActionPD.DEFECT), (ActionPD.DEFECT, ActionPD.DEFECT)): [0 for _ in range(episodes)],
    }

    agent_qvales_ps = [[0 for _ in range(episodes)] for _ in range(4)]
    agent_qvales_pd = [[0 for _ in range(episodes)] for _ in range(4)]

    recorded_qvalues_ps = [[[0 for _ in range(episodes)] for _ in range(4)] for _ in range(population)]
    recorded_qvalues_pd = [[[0 for _ in range(episodes)] for _ in range(4)] for _ in range(population)]

    agent_pd_actions_per_episode = [[0 for _ in range(episodes)] for _ in range(4)]
    percentage_of_states_per_episode = [[0 for _ in range(episodes)] for _ in range(4)]

    agent_chosen_switches_per_episode = [0 for _ in range(episodes)]
    agent_switches_per_episode = [0 for _ in range(episodes)]

    total_reward = [0 for _ in range(episodes)]

    # Fix Randoms
    # np.random.seed(0)

    # Global Q-Table Test
    # qtable_ps = np.zeros((2, 2))
    # qtable_pd = np.zeros((2, 2))
    # agents = [Agent(learning_rate, temperature, discount_rate, qtable_ps=qtable_ps, qtable_pd=qtable_pd) for _ in range(population)]

    agents = [Agent(learning_rate, temperature, discount_rate, delta_t=delta_t) for _ in range(population)]
    unpaired = list(range(population))

# Pair Agents
    pairs: tuple[int, int] = []
    while unpaired:
        i = unpaired.pop(np.random.randint(len(unpaired)))
        j = unpaired.pop(np.random.randint(len(unpaired)))
        pairs.append((i, j))

    probabilities_ps_defected = []
    probabilities_ps_cooperated = []
    probabilities_pd_defected = []
    probabilities_pd_cooperated = []
    max_probabilities_ps_defected = []
    max_probabilities_ps_cooperated = []
    max_probabilities_pd_defected = []
    max_probabilities_pd_cooperated = []
    new_probabilities_ps_defected = []
    new_probabilities_pd_defected = []
    new_probabilities_ps_cooperated = []
    new_probabilities_pd_cooperated = []
    new_max_probabilities_ps_defected = []
    new_max_probabilities_pd_defected = []
    new_max_probabilities_ps_cooperated = []
    new_max_probabilities_pd_cooperated = []
    strategies_ps = []
    strategies_pd = []
    new_strategies_ps = []
    new_strategies_pd = []

    for episode in range(episodes):
        # Record agent Q-Values
        for agent_idx in range(len(recorded_qvalues_ps)):
            for idx in range(len(recorded_qvalues_ps[agent_idx])):
                recorded_qvalues_ps[agent_idx][idx][episode] = agents[agent_idx].qtable_ps.ravel()[idx]

        for agent_idx in range(len(recorded_qvalues_pd)):
            for idx in range(len(recorded_qvalues_pd[agent_idx])):
                recorded_qvalues_pd[agent_idx][idx][episode] = agents[agent_idx].qtable_pd.ravel()[idx]

        trajectories = [[] for _ in range(population)]
        for round in range(rounds):

            # Partner Selection
            temp_pairs = []
            switch_leave_pool = []
            switch_stay_pool = []
            stay_pool = []

            for (i, j) in pairs:
                s_i = get_state(agents[j].last_action_pd)
                s_j = get_state(agents[i].last_action_pd)
                percentage_of_states_per_episode[s_i.value][episode] += 1
                percentage_of_states_per_episode[s_j.value][episode] += 1

                a_i = agents[i].get_action_ps(s_i)
                a_j = agents[j].get_action_ps(s_j)

                prob_ps_defected_i = boltzmann_exploration(agents[i].qtable_ps, State.PARTNER_DEFECTED, agents[i].t)
                prob_ps_cooperated_i = boltzmann_exploration(agents[i].qtable_ps, State.PARTNER_COOPERATED, agents[i].t)
                prob_ps_defected_j = boltzmann_exploration(agents[j].qtable_ps, State.PARTNER_DEFECTED, agents[j].t)
                prob_ps_cooperated_j = boltzmann_exploration(agents[j].qtable_ps, State.PARTNER_COOPERATED, agents[j].t)

                probabilities_ps_defected.append(prob_ps_defected_i[0])
                probabilities_ps_defected.append(prob_ps_defected_j[0])
                probabilities_ps_cooperated.append(prob_ps_cooperated_i[0])
                probabilities_ps_cooperated.append(prob_ps_cooperated_j[0])
                max_probabilities_ps_defected.append(max(prob_ps_defected_i))
                max_probabilities_ps_defected.append(max(prob_ps_defected_j))
                max_probabilities_ps_cooperated.append(max(prob_ps_cooperated_i))
                max_probabilities_ps_cooperated.append(max(prob_ps_cooperated_j))

                if a_i == ActionPS.LEAVE or a_j == ActionPS.LEAVE:
                    if a_i == ActionPS.LEAVE:
                        switch_leave_pool.append(i)
                        agents[i].last_action_ps = ActionPS.LEAVE
                        agents[i].last_result_ps = "split"
                    else:
                        switch_stay_pool.append(i)
                        agents[i].last_action_ps = ActionPS.STAY
                        agents[i].last_result_ps = "split"
                    if a_j == ActionPS.LEAVE:
                        switch_leave_pool.append(j)
                        agents[j].last_action_ps = ActionPS.LEAVE
                        agents[j].last_result_ps = "split"
                    else:
                        switch_stay_pool.append(j)
                        agents[j].last_action_ps = ActionPS.STAY
                        agents[j].last_result_ps = "split"
                    agent_switches_per_episode[episode] += 2
                    agent_chosen_switches_per_episode[episode] += int(a_i == ActionPS.LEAVE) + int(a_j == ActionPS.LEAVE)
                else:
                    stay_pool.append((i, j))
                    agents[i].last_action_ps = ActionPS.STAY
                    agents[i].last_result_ps = "stay"
                    agents[j].last_action_ps = ActionPS.STAY
                    agents[j].last_result_ps = "stay"

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
                if (rand_choice < prefer_same_pool) and i_pool:
                    j_pool = i_pool
                    j = j_pool.pop(np.random.randint(len(j_pool)))
                elif (rand_choice < prefer_same_pool + prefer_different_pool) and other_pool:
                    j_pool = other_pool
                    j = j_pool.pop(np.random.randint(len(j_pool)))
                else:
                    j = full_pool.pop(np.random.randint(len(full_pool)))
                    j_pool = switch_leave_pool if j in switch_leave_pool else switch_stay_pool
                    j_pool.remove(j)
                temp_pairs.append((i, j))

            # Pair remaining agents in switch_pool with each other
            while len(switch_leave_pool) > 1:
                i = switch_leave_pool.pop(np.random.randint(len(switch_leave_pool)))
                j = switch_leave_pool.pop(np.random.randint(len(switch_leave_pool)))
                temp_pairs.append((i, j))

            # Pair remaining agents in switched_pool with each other
            while len(switch_stay_pool) > 1:
                i = switch_stay_pool.pop(np.random.randint(len(switch_stay_pool)))
                j = switch_stay_pool.pop(np.random.randint(len(switch_stay_pool)))
                temp_pairs.append((i, j))

            # Combine new pairs with stay_pool pairs
            pairs = temp_pairs + stay_pool

            # Prisoner's Dilemma
            for (i, j) in pairs:
                strategy_i = agents[i].get_strategy_pd()
                strategy_j = agents[j].get_strategy_pd()
                if (strategy_i, strategy_j) in recorded_agent_strategy_pairings:
                    recorded_agent_strategy_pairings[(strategy_i, strategy_j)][episode] += 1
                elif (strategy_j, strategy_i) in recorded_agent_strategy_pairings:
                    recorded_agent_strategy_pairings[(strategy_j, strategy_i)][episode] += 1

                # If agent split, use random action
                if (agents[i].last_result_ps == "split" and agents[j].last_result_ps == "split") and np.random.rand() > know_fresh_agent:
                    s_i = State.PARTNER_COOPERATED if np.random.rand() < disposition else State.PARTNER_DEFECTED
                    s_j = State.PARTNER_COOPERATED if np.random.rand() < disposition else State.PARTNER_DEFECTED
                else:
                    s_i = get_state(agents[j].last_action_pd)
                    s_j = get_state(agents[i].last_action_pd)
                percentage_of_states_per_episode[(s_i.value) + 2][episode] += 1
                percentage_of_states_per_episode[(s_j.value) + 2][episode] += 1

                a_i = agents[i].get_action_pd(s_i)
                a_j = agents[j].get_action_pd(s_j)

                prob_pd_defected_i = boltzmann_exploration(agents[i].qtable_pd, State.PARTNER_DEFECTED, agents[i].t)
                prob_pd_cooperated_i = boltzmann_exploration(agents[i].qtable_pd, State.PARTNER_COOPERATED, agents[i].t)
                prob_pd_defected_j = boltzmann_exploration(agents[j].qtable_pd, State.PARTNER_DEFECTED, agents[j].t)
                prob_pd_cooperated_j = boltzmann_exploration(agents[j].qtable_pd, State.PARTNER_COOPERATED, agents[j].t)
                
                probabilities_pd_defected.append(prob_pd_defected_i[0])
                probabilities_pd_defected.append(prob_pd_defected_j[0])
                probabilities_pd_cooperated.append(prob_pd_cooperated_i[0])
                probabilities_pd_cooperated.append(prob_pd_cooperated_j[0])
                max_probabilities_pd_defected.append(max(prob_pd_defected_i))
                max_probabilities_pd_defected.append(max(prob_pd_defected_j))
                max_probabilities_pd_cooperated.append(max(prob_pd_cooperated_i))
                max_probabilities_pd_cooperated.append(max(prob_pd_cooperated_j))

                r_i, r_j = prisoners_dilemma(a_i, a_j)
                total_reward[episode] += r_i + r_j

                ns_i = get_state(a_j)
                ns_j = get_state(a_i)
                recorded_outcomes_pd[(a_i, a_j)][episode] += 1
                agents[i].last_action_pd = a_i
                agents[j].last_action_pd = a_j

                # Record Trajectories
                t = trajectories[i][round]
                trajectories[i][round] = (t[0], t[1], t[2], s_i, a_i, ns_i, r_i)
                t = trajectories[j][round]
                trajectories[j][round] = (t[0], t[1], t[2], s_j, a_j, ns_j, r_j)

                # Record Actions taken
                agent_pd_actions_per_episode[2 * (s_i.value - 2) + a_i.value][episode] += 1
                agent_pd_actions_per_episode[2 * (s_j.value - 2) + a_j.value][episode] += 1

        # print()
        
        for idx, agent in enumerate(agents):
            agent.train(trajectories[idx], learning_mode, last_trajectory=agent.last_trajectory)
            # agent.train(trajectories[idx], debug=(idx == 0))
        for idx, agent in enumerate(agents):
            agent.last_trajectory = trajectories[idx][-1]
        
        for idx in range(len(agent_qvales_ps)):
            agent_qvales_ps[idx][episode] = np.sum([agent.qtable_ps.ravel()[idx] for agent in agents]) / len(agents)

        for idx in range(len(agent_qvales_pd)):
            agent_qvales_pd[idx][episode] = np.sum([agent.qtable_pd.ravel()[idx] for agent in agents]) / len(agents)

        recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.DEFECT)][episode] /= (rounds * population / 2)
        recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.COOPERATE)][episode] /= (rounds * population / 2)
        recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.DEFECT)][episode] /= (rounds * population / 2)
        recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.COOPERATE)][episode] /= (rounds * population / 2)

        agent_pd_actions_per_episode[0][episode] /= (rounds * population)
        agent_pd_actions_per_episode[1][episode] /= (rounds * population)
        agent_pd_actions_per_episode[2][episode] /= (rounds * population)
        agent_pd_actions_per_episode[3][episode] /= (rounds * population)

        percentage_of_states_per_episode[0][episode] /= (rounds * population)
        percentage_of_states_per_episode[1][episode] /= (rounds * population)
        percentage_of_states_per_episode[2][episode] /= (rounds * population)
        percentage_of_states_per_episode[3][episode] /= (rounds * population)

        for idx, agent in enumerate(agents):
            strategies_ps.append(agent.get_strategy_ps())
            strategies_pd.append(agent.get_strategy_pd())

        for agent_trajectories in trajectories:
            for idx in range(rounds - 1):
                outcome = (agent_trajectories[idx][4], ActionPD.COOPERATE if agent_trajectories[idx][5] == State.PARTNER_COOPERATED else ActionPD.DEFECT)
                next_outcome = (agent_trajectories[idx + 1][4], ActionPD.COOPERATE if agent_trajectories[idx + 1][5] == State.PARTNER_COOPERATED else ActionPD.DEFECT)
                # if (outcome, next_outcome) in recorded_outcome_changes:
                recorded_outcome_changes[(outcome, next_outcome)][episode] += 1
                # else:
                #     recorded_outcome_changes[((outcome[1], outcome[0]), (next_outcome[1], next_outcome[0]))][episode] += 1

        mean_probabilities_ps_defected = np.mean(probabilities_ps_defected)
        mean_probabilities_ps_cooperated = np.mean(probabilities_ps_cooperated)
        mean_probabilities_pd_defected = np.mean(probabilities_pd_defected)
        mean_probabilities_pd_cooperated = np.mean(probabilities_pd_cooperated)
        mean_max_probabilities_ps_defected = np.mean(max_probabilities_ps_defected)
        mean_max_probabilities_ps_cooperated = np.mean(max_probabilities_ps_cooperated)
        mean_max_probabilities_pd_defected = np.mean(max_probabilities_pd_defected)
        mean_max_probabilities_pd_cooperated = np.mean(max_probabilities_pd_cooperated)
        
        new_probabilities_ps_defected.append(mean_probabilities_ps_defected)
        new_probabilities_ps_cooperated.append(mean_probabilities_ps_cooperated)
        new_probabilities_pd_defected.append(mean_probabilities_pd_defected)
        new_probabilities_pd_cooperated.append(mean_probabilities_pd_cooperated)
        new_max_probabilities_ps_defected.append(mean_max_probabilities_ps_defected)
        new_max_probabilities_ps_cooperated.append(mean_max_probabilities_ps_cooperated)
        new_max_probabilities_pd_defected.append(mean_max_probabilities_pd_defected)
        new_max_probabilities_pd_cooperated.append(mean_max_probabilities_pd_cooperated)
        new_strategies_ps.append(strategies_ps)
        new_strategies_pd.append(strategies_pd)
        
        probabilities_ps_defected = []
        probabilities_ps_cooperated = []
        probabilities_pd_defected = []
        probabilities_pd_cooperated = []        
        max_probabilities_ps_defected = []
        max_probabilities_ps_cooperated = []
        max_probabilities_pd_defected = []
        max_probabilities_pd_cooperated = []
        strategies_ps = []
        strategies_pd = []

    strat_always_leave = [sum([1 for strategy in strategies if strategy == StrategyPS.ALWAYS_LEAVE]) for strategies in new_strategies_ps]
    strat_out_for_tat = [sum([1 for strategy in strategies if strategy == StrategyPS.OUT_FOR_TAT]) for strategies in new_strategies_ps]
    strat_reverse_out_for_tat = [sum([1 for strategy in strategies if strategy == StrategyPS.REVERSE_OUT_FOR_TAT]) for strategies in new_strategies_ps]
    strat_always_stay = [sum([1 for strategy in strategies if strategy == StrategyPS.ALWAYS_STAY]) for strategies in new_strategies_ps]
    strat_always_defect = [sum([1 for strategy in strategies if strategy == StrategyPD.ALWAYS_DEFECT]) for strategies in new_strategies_pd]
    strat_tft = [sum([1 for strategy in strategies if strategy == StrategyPD.TFT]) for strategies in new_strategies_pd]
    strat_rtft = [sum([1 for strategy in strategies if strategy == StrategyPD.RTFT]) for strategies in new_strategies_pd]
    strat_always_cooperate = [sum([1 for strategy in strategies if strategy == StrategyPD.ALWAYS_COOPERATE]) for strategies in new_strategies_pd]

    # for agent_idx in range(recorded_qvalues_ps):
    #         for idx in range(recorded_qvalues_ps[agent_idx]):
    #             recorded_qvalues_ps[agent_idx][idx][episodes] = agent.qtable_ps.ravel()[idx]

    num_strategies_ps = [0 for _ in StrategyPS]
    num_strategies_pd = [0 for _ in StrategyPD]

    strategy_combinations = np.zeros((len(StrategyPS), len(StrategyPD)))

    # Determine Agent Strategies
    for idx, agent in enumerate(agents):
        strategy_ps = agent.get_strategy_ps()
        strategy_pd = agent.get_strategy_pd()
        
        num_strategies_ps[strategy_ps.value] += 1
        num_strategies_pd[strategy_pd.value] += 1
        strategy_combinations[strategy_ps.value, strategy_pd.value] += 1

        if do_plot:
            print("Agent %i) PS-Strategy: %s, PD-Strategy: %s" % 
                (idx, strategy_names[strategy_ps], strategy_names[strategy_pd]))
            print(agent.qtable_ps)
            print(agent.qtable_pd)

    results = {
        # "agents": agents,
        "recorded_outcomes_pd": recorded_outcomes_pd,
        "recorded_agent_strategy_pairings": recorded_agent_strategy_pairings,
        "recorded_outcome_changes": recorded_outcome_changes,
        "agent_qvales_ps": agent_qvales_ps,
        "agent_qvales_pd": agent_qvales_pd,
        "recorded_qvalues_ps": recorded_qvalues_ps,
        "recorded_qvalues_pd": recorded_qvalues_pd,
        "agent_pd_actions_per_episode": agent_pd_actions_per_episode,
        "percentage_of_states_per_episode": percentage_of_states_per_episode,
        "agent_chosen_switches_per_episode": agent_chosen_switches_per_episode,
        "agent_switches_per_episode": agent_switches_per_episode,
        "total_reward": total_reward,
        "ps_strategies": {
            StrategyPS.ALWAYS_LEAVE: strat_always_leave,
            StrategyPS.OUT_FOR_TAT: strat_out_for_tat,
            StrategyPS.REVERSE_OUT_FOR_TAT: strat_reverse_out_for_tat,
            StrategyPS.ALWAYS_STAY: strat_always_stay,
        },
        "pd_strategies": {
            StrategyPD.ALWAYS_DEFECT: strat_always_defect,
            StrategyPD.TFT: strat_tft,
            StrategyPD.RTFT: strat_rtft,
            StrategyPD.ALWAYS_COOPERATE: strat_always_cooperate,
        },
        "new_probabilities_ps_defected": new_probabilities_ps_defected,
        "new_probabilities_ps_cooperated": new_probabilities_ps_cooperated,
        "new_probabilities_pd_defected": new_probabilities_pd_defected,
        "new_probabilities_pd_cooperated": new_probabilities_pd_cooperated,
        "new_max_probabilities_ps_defected": new_max_probabilities_ps_defected,
        "new_max_probabilities_ps_cooperated": new_max_probabilities_ps_cooperated,
        "new_max_probabilities_pd_defected": new_max_probabilities_pd_defected,
        "new_max_probabilities_pd_cooperated": new_max_probabilities_pd_cooperated,
        "population": population,
        "rounds": rounds,
        "num_strategies_ps": num_strategies_ps,
        "num_strategies_pd": num_strategies_pd,
        "strategy_combinations": strategy_combinations,
    }

    if do_plot:
        plot_all(results)

    return results

def plot_all(results):
    plot_pd_outcomes(results)
    plot_strategies(results)
    plot_mean_probabilities(results)
    plot_rewards(results)
    plot_agent_switches_per_episode(results)
    plot_percentage_of_pd_actions_per_episode(results)
    plot_percentage_of_states_per_episode(results)
    plot_final_strategies(results)
    plot_strategy_combinations(results)
    plot_agent_strategy_pairings(results)
    plot_average_qvalues(results)
    plot_agents_qvalues(results)
    plot_outcome_changes(results)

def plot_pd_outcomes(results):
    recorded_outcomes_pd = results["recorded_outcomes_pd"]
    # Plot Prisoner's Dilemma Outcomes
    plt.plot(recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.DEFECT)], linewidth=1)
    plt.plot(recorded_outcomes_pd[(ActionPD.DEFECT, ActionPD.COOPERATE)], linewidth=1)
    plt.plot(recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.DEFECT)], linewidth=1)
    plt.plot(recorded_outcomes_pd[(ActionPD.COOPERATE, ActionPD.COOPERATE)], linewidth=1)
    
    plt.title("Percentage of Prisoner's Dilemma Outcomes Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Percentage of Outcomes')
    plt.legend(["(D, D)", "(D, C)", "(C, D)", "(C, C)", "Total Reward"])
    plt.show()


def plot_strategies(results, do_ps=True, do_pd=True):
    strat_always_leave = results["ps_strategies"][StrategyPS.ALWAYS_LEAVE]
    strat_out_for_tat = results["ps_strategies"][StrategyPS.OUT_FOR_TAT]
    strat_reverse_out_for_tat = results["ps_strategies"][StrategyPS.REVERSE_OUT_FOR_TAT]
    strat_always_stay = results["ps_strategies"][StrategyPS.ALWAYS_STAY]
    strat_always_defect = results["pd_strategies"][StrategyPD.ALWAYS_DEFECT]
    strat_tft = results["pd_strategies"][StrategyPD.TFT]
    strat_rtft = results["pd_strategies"][StrategyPD.RTFT]
    strat_always_cooperate = results["pd_strategies"][StrategyPD.ALWAYS_COOPERATE]

    if do_ps:
        plt.plot(strat_always_leave, linewidth=1)
        plt.plot(strat_out_for_tat, linewidth=1)
        plt.plot(strat_reverse_out_for_tat, linewidth=1)
        plt.plot(strat_always_stay, linewidth=1)

        plt.title("Number of Partner Selection Strategies Per Episode")
        plt.xlabel('Episode')
        plt.ylabel('Number of Agents')
        plt.legend(["Always Leave", "Out For Tat", "Reverse Out For Tat", "Always Stay"])
        plt.show()

    if do_pd:
        plt.plot(strat_always_defect)
        plt.plot(strat_tft)
        plt.plot(strat_rtft)
        plt.plot(strat_always_cooperate)

        plt.title("Number of Prisoner's Dilemma Strategies Per Episode")
        plt.xlabel('Episode')
        plt.ylabel('Number of Agents')
        plt.legend(["Always Defect", "Tit For Tat", "Reverse Tit For Tat", "Always Cooperate"])
        plt.show()

def plot_mean_probabilities(results, do_ps=True, do_pd=True):
    new_probabilities_pd_defected = results["new_probabilities_pd_defected"]
    new_probabilities_pd_cooperated = results["new_probabilities_pd_cooperated"]
    new_probabilities_ps_defected = results["new_probabilities_ps_defected"]
    new_probabilities_ps_cooperated = results["new_probabilities_ps_cooperated"]

    if do_pd:
        # Plot Mean Probabilities of Defection and Cooperation
        plt.title("Probabilities of defection given state in Prisoner's Dilemma Stage")
        plt.plot(new_probabilities_pd_defected)
        plt.plot(new_probabilities_pd_cooperated)
        plt.xlabel("Episodes")
        plt.legend(["Partner Defected", "Partner Cooperated"])
        plt.ylabel("Probability of Defection")
        plt.show()

    if do_ps:
        plt.title("Probabilities of leaving given state in Partner Selection Stage")
        plt.plot(new_probabilities_ps_defected)
        plt.plot(new_probabilities_ps_cooperated)
        plt.xlabel("Episodes")
        plt.legend(["Partner Defected", "Partner Cooperated"])
        plt.ylabel("Probability of Leaving")
        plt.show()
        


def plot_rewards(results):
    total_reward = results["total_reward"]
    population = results["population"]
    rounds = results["rounds"]
    # Plot Total Rewards
    plt.plot(total_reward, linewidth=3)

    plt.title("Total Reward Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.ylim(0, 6 * population * rounds / 2)
    plt.show()

def plot_agent_switches_per_episode(results):
    agent_chosen_switches_per_episode = results["agent_chosen_switches_per_episode"]
    agent_switches_per_episode = results["agent_switches_per_episode"]
    population = results["population"]
    rounds = results["rounds"]
    
    # Plot Agent/Pair Switches Per Episode
    plt.plot(np.divide(agent_chosen_switches_per_episode, rounds * population))
    plt.plot(np.divide(agent_switches_per_episode, rounds * population))

    plt.title("Percentage of Switches Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Percentage of Agents')
    plt.legend(["Agents Who Chose to Switch Partners", "Agents Who Switched Partners"])
    plt.show()

def plot_percentage_of_pd_actions_per_episode(results):
    agent_pd_actions_per_episode = results["agent_pd_actions_per_episode"]
    # Plot Percentage of Agent PD Actions Per Episode Given State
    plt.plot(agent_pd_actions_per_episode[0])
    plt.plot(agent_pd_actions_per_episode[1])
    plt.plot(agent_pd_actions_per_episode[2])
    plt.plot(agent_pd_actions_per_episode[3])
    
    plt.title("Percentage of PD Actions Per Episode Given State")
    plt.xlabel('Episode')
    plt.ylabel('Percentage of Agents')
    plt.legend(["Defected Given Parter Previously Defected", "Cooperated Given Parter Previously Defected", 
                "Defected Given Parter Previously Cooperated", "Cooperated Given Parter Previously Cooperated"],
                loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.show()

def plot_percentage_of_states_per_episode(results, do_ps=True, do_pd=True):
    percentage_of_states_per_episode = results["percentage_of_states_per_episode"]
    
    if do_ps:
        # Plot Percentage of Agent States Per Episode
        plt.subplot(211)
        plt.plot(percentage_of_states_per_episode[0])
        plt.plot(percentage_of_states_per_episode[1])
        
        plt.title("Percentage of Partner Selection Agent States Per Episode")
        plt.xlabel('Episode')
        plt.ylabel('Percentage of States')
        plt.legend(["Partner Selection Where Parter Previously Defected", "Partner Selection Where Parter Previously Cooperated"], 
                    loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.show()

    if do_pd:
        plt.subplot(212)
        plt.plot(percentage_of_states_per_episode[2], linestyle='dotted')
        plt.plot(percentage_of_states_per_episode[3], linestyle='dotted')
        
        plt.title("Percentage of Prisoner's Dilemma Agent States Per Episode")
        plt.xlabel('Episode')
        plt.ylabel('Percentage of States')
        plt.legend(["Prisoner's Dilemma Where Parter Previously Defected", "Prisoner's Dilemma Where Parter Previously Cooperated"], 
                    loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.show()

def plot_final_strategies(results, do_ps=True, do_pd=True):
    num_strategies_ps = results["num_strategies_ps"]
    num_strategies_pd = results["num_strategies_pd"]
    population = results["population"]

    if do_ps:
        plt.subplot(211)
        ps_colors = [strategy_colors[strategy] for strategy in StrategyPS]
        plt.bar([strategy_names[strategy] for strategy in StrategyPS], num_strategies_ps, color=ps_colors)
        plt.title('PS-Strategies')
        plt.xlabel('Strategy')
        plt.ylabel('Number of Agents')
        plt.ylim(0, population)
        plt.show()

    if do_pd:
        plt.subplot(212)
        pd_colors = [strategy_colors[strategy] for strategy in StrategyPD]
        plt.bar([strategy_names[strategy] for strategy in StrategyPD], num_strategies_pd, color=pd_colors)
        plt.title('PD-Strategies')
        plt.xlabel('Strategy')
        plt.ylabel('Number of Agents')
        plt.ylim(0, population)
        plt.show()


def plot_strategy_combinations(results):
    strategy_combinations = results["strategy_combinations"]
    population = results["population"]
    
    # Plot agent partner selection and prisoner's dilemma strategy combinations
    combination_indices = [i for i in range(len(strategy_combinations.ravel()))]
    ps_combination_colors = np.repeat([strategy_colors[strategy] for strategy in StrategyPS], 5)
    pd_combination_colors = np.tile([strategy_colors[strategy] for strategy in StrategyPD], 5)
    plt.bar(combination_indices, strategy_combinations.ravel() / 2.0, color=ps_combination_colors)
    plt.bar(combination_indices, strategy_combinations.ravel() / 2.0, bottom=(strategy_combinations.ravel() / 2.0), color=pd_combination_colors)
    plt.title('Final Strategy Combinations')
    plt.xlabel('Strategy Combination')
    plt.ylabel('Number of Agents')
    # plt.ylim(0, population)
    plt.legend(
        [plt.Rectangle((0, 0), 1, 1, color=value) for key, value in strategy_colors.items()],
        [strategy_names[key] for key, value in strategy_colors.items()],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=5,
    )
    plt.show()

def plot_agent_strategy_pairings(results):
    recorded_agent_strategy_pairings = results["recorded_agent_strategy_pairings"]
    # Plot Prisoner's Dilemma Strategy Pairings
    for pairing, values in recorded_agent_strategy_pairings.items():
        plt.plot(values, '-', color=strategy_colors[pairing[0]], linewidth=3)
        plt.plot(values, '--', color=strategy_colors[pairing[1]], linewidth=3)
    
    plt.title("Prisoner's Dilemma Strategy Pairings Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Number of Pairings')
    plt.legend(
        [plt.Rectangle((0, 0), 1, 1, color=strategy_colors[strategy]) for strategy in StrategyPD],
        [strategy_names[strategy] for strategy in StrategyPD],
        loc='upper center',
        bbox_to_anchor=(0.5, 1.35),
        ncol=5,
    )
    plt.show()


def plot_average_qvalues(results, do_ps=True, do_pd=True):
    agent_qvales_ps = results["agent_qvales_ps"]
    agent_qvales_pd = results["agent_qvales_pd"]
    population = results["population"]

    if do_ps:
        # Plot Average Partner Selection Q-Values in each episode
        plt.plot(agent_qvales_ps[0])
        plt.plot(agent_qvales_ps[1])
        plt.plot(agent_qvales_ps[2], linestyle='dotted')
        plt.plot(agent_qvales_ps[3], linestyle='dotted')
        
        plt.title("Partner Selection Q-Values in each episode")
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.legend(["Leave Given Parter Previously Defected", "Stay Given Parter Previously Defected", 
                    "Leave Given Parter Previously Cooperated", "Stay Given Parter Previously Cooperated"],
                    loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.show()

    if do_pd:
        # Plot Average Prisoner's Dilemma Q-Values in each episode
        plt.plot(agent_qvales_pd[0])
        plt.plot(agent_qvales_pd[1])
        plt.plot(agent_qvales_pd[2], linestyle='dotted')
        plt.plot(agent_qvales_pd[3], linestyle='dotted')
        
        plt.title("Prisoner's Dilemma Q-Values in each episode")
        plt.xlabel('Episode')
        plt.ylabel('Q-Value')
        plt.legend(["Defect Given Parter Previously Defected", "Cooperate Given Parter Previously Defected", 
                    "Defect Given Parter Previously Cooperated", "Cooperate Given Parter Previously Cooperated"],
                    loc='center left', bbox_to_anchor=(1.04, 0.5))
        plt.show()

def plot_agents_qvalues(results, do_ps=False, do_pd=True):
    recorded_qvalues_ps = results["recorded_qvalues_ps"]
    recorded_qvalues_pd = results["recorded_qvalues_pd"]
    population = results["population"]
    
    # Plot Agents Partner Selection Q-Values in each episode
    if do_ps:
        for idx in range(population):
            plt.subplot(211)
            plt.plot(recorded_qvalues_ps[idx][0])
            plt.plot(recorded_qvalues_ps[idx][1])
            plt.plot(recorded_qvalues_ps[idx][2], linestyle='dotted')
            plt.plot(recorded_qvalues_ps[idx][3], linestyle='dotted')
            
            plt.title("Agent " + str(idx) + " Partner Selection Q-Values in each episode")
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend(["Leave Given Parter Previously Defected", "Stay Given Parter Previously Defected", 
                        "Leave Given Parter Previously Cooperated", "Stay Given Parter Previously Cooperated"],
                        loc='center left', bbox_to_anchor=(1.04, 0.5))
            plt.show()

            plt.subplot(212)
            plt.plot(recorded_qvalues_pd[idx][0])
            plt.plot(recorded_qvalues_pd[idx][1])
            plt.plot(recorded_qvalues_pd[idx][2], linestyle='dotted')
            plt.plot(recorded_qvalues_pd[idx][3], linestyle='dotted')
            
            plt.title("Agent " + str(idx) + " Prisoner's Dilemma Q-Values in each episode")
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend(["Defect Given Parter Previously Defected", "Cooperate Given Parter Previously Defected", 
                        "Defect Given Parter Previously Cooperated", "Cooperate Given Parter Previously Cooperated"],
                        loc='center left', bbox_to_anchor=(1.04, 0.5))
            plt.show()

    # Plot Agents Prisoner's Dilemma Q-Values in each episode
    if do_pd:
        for idx in range(population):
            plt.subplot(211)
            plt.plot(recorded_qvalues_ps[idx][0])
            plt.plot(recorded_qvalues_ps[idx][1])
            plt.plot(recorded_qvalues_ps[idx][2], linestyle='dotted')
            plt.plot(recorded_qvalues_ps[idx][3], linestyle='dotted')
            
            plt.title("Agent " + str(idx) + " Partner Selection Q-Values in each episode")
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend(["Leave Given Parter Previously Defected", "Stay Given Parter Previously Defected", 
                        "Leave Given Parter Previously Cooperated", "Stay Given Parter Previously Cooperated"],
                        loc='center left', bbox_to_anchor=(1.04, 0.5))
            plt.show()

            plt.subplot(212)
            plt.plot(recorded_qvalues_pd[idx][0])
            plt.plot(recorded_qvalues_pd[idx][1])
            plt.plot(recorded_qvalues_pd[idx][2], linestyle='dotted')
            plt.plot(recorded_qvalues_pd[idx][3], linestyle='dotted')
            
            plt.title("Agent " + str(idx) + " Prisoner's Dilemma Q-Values in each episode")
            plt.xlabel('Episode')
            plt.ylabel('Q-Value')
            plt.legend(["Defect Given Parter Previously Defected", "Cooperate Given Parter Previously Defected", 
                        "Defect Given Parter Previously Cooperated", "Cooperate Given Parter Previously Cooperated"],
                        loc='center left', bbox_to_anchor=(1.04, 0.5))
            plt.show()

def plot_outcome_changes(results):
    recorded_outcome_changes = results["recorded_outcome_changes"]
    rounds = results["rounds"]
    population = results["population"]
    # plot recorded outcome changes
    outcome_changes_legend = []
    for keys, changes in recorded_outcome_changes.items():
        if keys[0] == (ActionPD.COOPERATE, ActionPD.COOPERATE):
            style = 'solid'
        elif keys[0] == (ActionPD.COOPERATE, ActionPD.DEFECT):
            style = 'dotted'
        elif keys[0] == (ActionPD.DEFECT, ActionPD.COOPERATE):
            style = 'dashed'
        elif keys[0] == (ActionPD.DEFECT, ActionPD.DEFECT):
            style = 'dashdot'

        if keys[1] == (ActionPD.COOPERATE, ActionPD.COOPERATE):
            color = 'red'
        elif keys[1] == (ActionPD.COOPERATE, ActionPD.DEFECT):
            color = 'green'
        elif keys[1] == (ActionPD.DEFECT, ActionPD.COOPERATE):
            color = 'orange'
        elif keys[1] == (ActionPD.DEFECT, ActionPD.DEFECT):
            color = 'blue'
        
        changes = np.divide(changes, rounds * population)

        plt.plot(changes, linestyle=style, color=color)
        outcome_changes_legend.append("(%s, %s) -> (%s, %s)" % (keys[0][0].name, keys[0][1].name, keys[1][0].name, keys[1][1].name))
    


    # Plot the changes in outcomes for prisoner's dilemma games per episode
    plt.title("Prisoner's Dilemma Outcome Changes Per Episode")
    plt.xlabel('Episode')
    plt.ylabel('Changes')
    plt.legend(outcome_changes_legend, loc='center left', bbox_to_anchor=(1.04, 0.5))
    plt.show()

# TODO PLOT REWARDS FOR EACH STRATEGY

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
    learning_type = "q_learning"

    base_params = {"population": 20,   # Agent Population Size (Must be a multiple of 2)
        "rounds": 20,            # Rounds per Episode
        "episodes": 4000,        # Number of Episodes
        "learning_rate": 0.05,   # Alpha (Learning Rate)
        "temperature": 85,       # Starting Boltzmann Temperature 
        "discount_rate": 0.99,   # Gamma (Discount Rate)
        "delta_t": 0.99,         # Boltzmann Temperature Decay Rate
        "disposition": 0.0,      # Disposition to Assume Cooperation
        "know_fresh_agent": 0.0, # Probability of Knowing Fresh Agent's Previous Action
        "prefer_same_pool": 0.0, # Probability of Choosing Same Pool Partner
        "prefer_different_pool": 0.0, # Probability of Choosing Different Pool Partner
        "learning_mode": learning_type, # Learning Mode (q_learning or sarsa)
        "do_plot": False}         # Plot Results

    tests = {
        "disposition": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    }

    reps = 5

    base_params["learning_mode"] = learning_type
    params = base_params.copy()

    all_results_reps = {}
    all_results_average = {}

    for param_name, param_values in tests.items():
        param_results_reps = {}
        param_results_avg = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            params = base_params.copy()
            params[param_name] = param_value
            output = []
            for _ in range(reps):
                output.append(sdoo(**params))
            param_results_reps[param_value] = {"params": params.copy(), "output": output}
            output_avg = average_results(output)
            param_results_avg[param_value] = {"params": params.copy(), "output": output_avg}
        with open(f'{param_name}_{learning_type}_reps.pkl', 'wb') as f:
            pickle.dump(param_results_reps, f)
        with open(f'{param_name}_{learning_type}_avg.pkl', 'wb') as f:
            pickle.dump(param_results_avg, f)
        all_results_reps[param_name] = param_results_reps
        all_results_average[param_name] = param_results_avg

def main1_5():
    with open('temperature_q_learning_reps.pkl', 'rb') as f:
        temperature_q_learning_reps = pickle.load(f)
    temperature_q_learning_avg = {}
    for param_value in temperature_q_learning_reps.keys():
        params = temperature_q_learning_reps[param_value]["params"]
        output = temperature_q_learning_reps[param_value]["output"]
        output_avg = average_results(output)
        temperature_q_learning_avg[param_value] = {"params": params.copy(), "output": output_avg}
    with open('temperature_q_learning_avg.pkl', 'wb') as f:
        pickle.dump(temperature_q_learning_avg, f)

def main2():
    learning_type = "q_learning"

    base_params = {"population": 20,   # Agent Population Size (Must be a multiple of 2)
        "rounds": 20,            # Rounds per Episode
        "episodes": 4000,        # Number of Episodes
        "learning_rate": 0.05,   # Alpha (Learning Rate)
        "temperature": 85,       # Starting Boltzmann Temperature 
        "discount_rate": 0.99,   # Gamma (Discount Rate)
        "delta_t": 0.99,         # Boltzmann Temperature Decay Rate
        "disposition": 0.0,      # Disposition to Assume Cooperation
        "know_fresh_agent": 0.0, # Probability of Knowing Fresh Agent's Previous Action
        "prefer_same_pool": 0.0, # Probability of Choosing Same Pool Partner
        "prefer_different_pool": 0.0, # Probability of Choosing Different Pool Partner
        "learning_mode": learning_type, # Learning Mode (q_learning or sarsa)
        "do_plot": False}         # Plot Results

    tests = {
        "discount_rate": [0.995, 0.99, 0.985, 0.98, 0.97, 0.95, 0.9, 0.8],
    }

    reps = 5

    base_params["learning_mode"] = learning_type
    params = base_params.copy()

    all_results_reps = {}
    all_results_average = {}

    for param_name, param_values in tests.items():
        param_results_reps = {}
        param_results_avg = {}
        for param_value in param_values:
            print(f"Testing {param_name} = {param_value}")
            params = base_params.copy()
            params[param_name] = param_value
            output = []
            for _ in range(reps):
                output.append(sdoo(**params))
            param_results_reps[param_value] = {"params": params.copy(), "output": output}
            output_avg = average_results(output)
            param_results_avg[param_value] = {"params": params.copy(), "output": output_avg}
        with open(f'{param_name}_{learning_type}_reps.pkl', 'wb') as f:
            pickle.dump(param_results_reps, f)
        with open(f'{param_name}_{learning_type}_avg.pkl', 'wb') as f:
            pickle.dump(param_results_avg, f)
        all_results_reps[param_name] = param_results_reps
        all_results_average[param_name] = param_results_avg

def main3():
    learning_type = "q_learning"
    policy_mode = "epsilon_greedy"
    episodes = 6000

    base_params = {"population": 20,   # Agent Population Size (Must be a multiple of 2)
        "rounds": 20,            # Rounds per Episode
        "learning_rate": 0.05,   # Alpha (Learning Rate)
        "temperature": 301,       # Starting Boltzmann Temperature 
        "discount_rate": 0.99,   # Gamma (Discount Rate)
        "delta_t": 0.1,         # Boltzmann Temperature Decay Rate
        "decay_type": "linear", # Decay Type (exponential or linear)
        "disposition": 0.0,      # Disposition to Assume Cooperation
        "know_fresh_agent": 1.0, # Probability of Knowing Fresh Agent's Previous Action
        "prefer_same_pool": 0.0, # Probability of Choosing Same Pool Partner
        "prefer_different_pool": 0.0, # Probability of Choosing Different Pool Partner
        "learning_mode": learning_type, # Learning Mode (q_learning or sarsa)
        "policy_mode": policy_mode, # Policy Mode (epsilon_greedy or boltzmann)
        }
    

    tests = {
        "discount_rate": [0.99],
    }

    reps = 5

    base_params["learning_mode"] = learning_type
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
            param_results_reps[param_value] = {"params": params.copy(), "output": output}
            output_avg = average_results(output)
            param_results_avg[param_value] = {"params": params.copy(), "output": output_avg}
        with open(f'reps/reps_{stts}.pkl', 'wb') as f:
            pickle.dump(param_results_reps, f)
        with open(f'avg_{stts}.pkl', 'wb') as f:
            pickle.dump(param_results_avg, f)
        all_results_reps[param_name] = param_results_reps
        all_results_average[param_name] = param_results_avg

if __name__ == "__main__":
    main3()