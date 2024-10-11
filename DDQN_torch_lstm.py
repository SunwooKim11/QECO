import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from torch.cuda.amp import GradScaler, autocast
import gc

class DuelingDoubleDeepQNetwork(nn.Module):
    def __init__(self, n_actions, n_features, n_lstm_features, n_time, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.99, replace_target_iter=200, memory_size=500,
                 batch_size=32, e_greedy_increment=0.00025, n_lstm_step=10, dueling=True,
                 double_q=True, hidden_units_l1=20, N_lstm=20, doLC=False):

        super(DuelingDoubleDeepQNetwork, self).__init__()
        self.doLC = doLC
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.hidden_units_l1 = hidden_units_l1

        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        self.eval_net = self._build_net().to(self.device)
        self.target_net = self._build_net().to(self.device)
        self.target_net.load_state_dict(self.eval_net.state_dict())

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()
        self.scaler = GradScaler()

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

    def _build_net(self):
        class Net(nn.Module):
            def __init__(self, dueling, hidden_units_l1, n_lstm, n_features, n_actions, n_lstm_state):
                super(Net, self).__init__()
                self.dueling = dueling

                self.lstm = nn.LSTM(input_size=n_lstm_state, hidden_size=n_lstm, batch_first=True)
                # self.lstm = nn.GRU(input_size=n_lstm_state, hidden_size=n_lstm, batch_first=True)

                self.fc1 = nn.Linear(n_lstm + n_features, hidden_units_l1)
                self.fc2 = nn.Linear(hidden_units_l1, hidden_units_l1)

                if self.dueling:
                    self.value_fc = nn.Linear(hidden_units_l1, 1)
                    self.advantage_fc = nn.Linear(hidden_units_l1, n_actions)
                else:
                    self.fc3 = nn.Linear(hidden_units_l1, n_actions)

            def forward(self, s, lstm_s):
                lstm_out, _ = self.lstm(lstm_s)
                lstm_out = lstm_out[:, -1, :]

                combined = torch.cat((lstm_out, s), dim=1)
                l1 = torch.relu(self.fc1(combined))
                l2 = torch.relu(self.fc2(l1))

                if self.dueling:
                    value = self.value_fc(l2)
                    advantage = self.advantage_fc(l2)
                    out = value + (advantage - advantage.mean(dim=1, keepdim=True))
                else:
                    out = self.fc3(l2)
                return out

        return Net(self.dueling, self.hidden_units_l1, self.N_lstm, self.n_features, self.n_actions, self.n_lstm_state)

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        if self.doLC:
            return 0

        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0).to(self.device)
        if np.random.uniform() < self.epsilon:
            lstm_observation = torch.tensor(np.array(self.lstm_history), dtype=torch.float).unsqueeze(0).to(self.device)
            with torch.no_grad():
                actions_value = self.eval_net(observation, lstm_observation)
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = torch.argmax(actions_value, dim=1).item()
        else:
            if np.random.randint(0, 100) < 25:
                action = np.random.randint(1, self.n_actions)
            else:
                action = 0

        return action

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            self.learn_step_counter += 1
            print('\ntarget_params_replaced')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])

        for i in range(len(sample_index)):
            for j in range(self.n_lstm_step):
                lstm_batch_memory[i, j, :] = self.memory[sample_index[i] + j,
                                                         self.n_features + 1 + 1 + self.n_features:]

        batch_memory = torch.tensor(batch_memory, dtype=torch.float).to(self.device)
        lstm_batch_memory = torch.tensor(lstm_batch_memory, dtype=torch.float).to(self.device)

        with autocast():
            q_next = self.target_net(batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:])
            q_eval4next = self.eval_net(batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:])

            q_eval = self.eval_net(batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state])

            q_target = q_eval.clone().detach()
            batch_index = np.arange(self.batch_size, dtype=np.int32)
            eval_act_index = batch_memory[:, self.n_features].long()
            reward = batch_memory[:, self.n_features + 1]

            if self.double_q:
                max_act4next = torch.argmax(q_eval4next, dim=1)
                selected_q_next = q_next[batch_index, max_act4next]
            else:
                selected_q_next, _ = torch.max(q_next, dim=1)

            q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

            loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        del batch_memory, lstm_batch_memory, q_next, q_eval4next, q_eval, q_target, loss
        gc.collect()
        torch.cuda.empty_cache()

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]

        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]

        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    def Initialize(self, iot):
        latest_model_path = f"./models/500/{iot}_X_model.pth"
        self.load_state_dict(torch.load(latest_model_path))

    def save_model(self, iot):
        model_path = f"./models/500/{iot}_X_model.pth"
        torch.save(self.state_dict(), model_path)
