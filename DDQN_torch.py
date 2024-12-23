import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class DuelingDoubleDeepQNetwork(nn.Module):
    # - 네트워크의 파라미터들과 함께 여러 중요한 변수를 초기화합니다.
    # - 예를 들어, n_actions는 가능한 행동의 수, n_features는 상태 특성의 수, learning_rate는 학습률 등입니다.
    # - self.memory는 경험 재생 메모리를 초기화합니다. 여기에는 각 경험이 저장되며, 학습 시 무작위로 샘플링하여 사용됩니다.
    # - _build_net() 함수를 호출하여 신경망을 구축합니다.
    # - 옵티마이저와 손실 함수를 정의합니다.
    
    def __init__(self, n_actions, n_features, n_lstm_features, n_time, learning_rate=0.01,
                 reward_decay=0.9, e_greedy=0.99, replace_target_iter=200, memory_size=500,
                 batch_size=32, e_greedy_increment=0.00025, n_lstm_step=10, dueling=True,
                 double_q=True, hidden_units_l1=20, N_lstm=20):

        super(DuelingDoubleDeepQNetwork, self).__init__()

        self.n_actions = n_actions  # self.n_actions = self.n_component + 1
        self.n_features = n_features # observation과 같음, 8임
        self.n_time = n_time
        self.lr = learning_rate
        self.gamma = reward_decay # 각 rewturn의 계수
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size #500
        self.batch_size = batch_size # 32
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.hidden_units_l1 = hidden_units_l1

        # lstm
        self.N_lstm = N_lstm # 20 -> hidden size?
        self.n_lstm_step = n_lstm_step # 10
        self.n_lstm_state = n_lstm_features # = self.n_edge -> input size
        # 모든 EN의 active 큐 수를 나타내는 배열이므로 ex) [3, 2, 2, 4, 1], EN이 5개일 때
        # s_, lstm_s_ -> s(t+1)
        # np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1
                                + self.n_features + self.n_lstm_state + self.n_lstm_state))

        self._build_net()

        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()

        # lstm history를 양방향 큐로
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = list()

    def _build_net(self):
        # 해당 네트워크를 bulid 할떼 파라미터(세타 E, T)가 생성되고, 이후 코드 진행하면서 학습
        # Build the neural network model

        hidden_units_l1 = self.hidden_units_l1
        N_lstm = self.N_lstm

        # torch.nn.LSTM(input_size=5, hidden_size=20, num_layers=1, bias=True, batch_first=True)
        self.lstm_dnn = nn.LSTM(self.n_lstm_state, N_lstm, batch_first=True)

        # Common layers
        self.fc1 = nn.Linear(N_lstm + self.n_features, hidden_units_l1)
        self.fc2 = nn.Linear(hidden_units_l1, hidden_units_l1)
        # Dueling 할지 DQN할지 선택할 수 있음.
        if self.dueling:
            # Dueling DQN
            # Value stream
            self.value = nn.Linear(hidden_units_l1, 1)
            # Advantage stream
            self.advantage = nn.Linear(hidden_units_l1, self.n_actions)
        else:
            self.q = nn.Linear(hidden_units_l1, self.n_actions)

    def forward(self, s, lstm_s):
        # 신경망의 입력으로 상태(s)와 LSTM의 상태(lstm_s)를 받아 최종 Q 값을 계산하여 반환합니다.
        # Dueling 구조를 사용하는 경우, 가치와 이점을 분리하여 계산한 뒤, 이를 합하여 최종 Q 값을 도출합니다.
        # Forward pass of the network

        lstm_output, _ = self.lstm_dnn(lstm_s)
        lstm_output_reduced = lstm_output[:, -1, :]

        x = torch.cat((lstm_output_reduced, s), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        if self.dueling:
            value = self.value(x)
            advantage = self.advantage(x)
            q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            q = self.q(x)

        return q

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        # - 행동(a), 보상(r), 새로운 상태(s_) 등의 경험을 메모리에 저장합니다. 이 정보는 학습 시 사용됩니다.

        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_)) # experience임 . Alogrithm2 Line 10
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        # - LSTM의 상태를 업데이트합니다. 이는 순차적 정보를 처리하기 위해 사용됩니다.
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        # - 현재 관찰(observation)을 바탕으로 행동을 선택합니다.
        # - 엡실론-탐욕(Epsilon-Greedy) 정책을 사용하여 대부분 최적의 행동을 선택하지만, 때때로 탐험을 위해 무작위 행동을 선택합니다.

        observation = torch.tensor(observation, dtype=torch.float).unsqueeze(0)

        if np.random.uniform() < self.epsilon:
            lstm_observation = torch.tensor(np.array(self.lstm_history), dtype=torch.float).unsqueeze(0)
            actions_value = self.forward(observation, lstm_observation)
            self.store_q_value.append({'observation': observation, 'q_value': actions_value})

            action = torch.argmax(actions_value, dim=1).item()
        else:
            if np.random.randint(0, 100) < 25:
                action = np.random.randint(1, self.n_actions)
            else:
                action = 0

        return action

    def learn(self):
        # - 메모리에서 무작위로 샘플링한 배치를 사용하여 네트워크를 학습합니다.
        # - DDQN의 경우, 타깃 Q 값 계산 시 평가 네트워크의 행동을 선택하고 타깃 네트워크로부터 해당 Q 값을 가져옵니다.
        # - 손실을 계산한 뒤, 역전파를 통해 네트워크의 가중치를 업데이트합니다.
        if self.learn_step_counter % self.replace_target_iter == 0:
            # No target network in PyTorch, this step can be omitted.
            print('\ntarget_params_replaced')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            # (batch_size=32,)크기의 0~memory_countr-10(n_lstm_step) 범위의 rand int 배열
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)
        # batch size 만큼. replay mem에서 s, [a, r], s_을 뽑아서 저장
        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])

        for i in range(len(sample_index)):
            for j in range(self.n_lstm_step):
                # lstm_s, lstm_s_ memory에 저장.
                lstm_batch_memory[i, j, :] = self.memory[sample_index[i] + j, self.n_features + 1 + 1 + self.n_features:]

        batch_memory = torch.tensor(batch_memory, dtype=torch.float)
        lstm_batch_memory = torch.tensor(lstm_batch_memory, dtype=torch.float)

        # QT(s(n+1)), QE(s(n+1))/ lstm 셀 10 개를 이어붙인 걸 lstm_dnn에 통과하네?
        q_next, q_eval4next = self.forward(batch_memory[:, -self.n_features:], lstm_batch_memory[:, :, self.n_lstm_state:]) # ??

        q_eval = self.forward(batch_memory[:, :self.n_features], lstm_batch_memory[:, :, :self.n_lstm_state])

        q_target = q_eval.clone().detach()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].long()
        reward = batch_memory[:, self.n_features + 1]

        # ?
        if self.double_q:
            max_act4next = torch.argmax(q_eval4next, dim=1)  # 수식 29
            selected_q_next = q_next[batch_index, max_act4next] # QT(s(n+1), a~; 세타Tn)
        else:
            selected_q_next, _ = torch.max(q_next, dim=1)
        # Algorithm2 Line 14, 수식 28
        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # epsilon 업데이트
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 각 에피소드에서의 보상, 행동, 지연, 에너지 사용량을 저장합니다. 이 데이터는 학습 과정을 모니터링하거나 분석할 때 유용하게 사용됩니다.
    
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
        
        # 모델을 초기화하거나 저장.
        # - 특정 IoT 장치에 대해 모델을 초기화하거나 저장합니다. 모델의 상태는 .pth 파일로 저장됩니다.
        # - 이 클래스는 신경망이 어떻게 구성되고, 데이터가 어떻게 처리되며, 학습이 어떻게 이루어지는지를 포함한 DDQN의 전체적인 프로세스를 정의합니다.
        # - Dueling과 Double Q-Learning 기법을 사용하여 보다 정교한 학습이 가능하도록 설계되었습니다.

    def Initialize(self, iot):
        latest_model_path = f"./models/500/{iot}_X_model.pth"
        self.load_state_dict(torch.load(latest_model_path))

    def save_model(self, iot):
        model_path = f"./models/500/{iot}_X_model.pth"
        torch.save(self.state_dict(), model_path)
