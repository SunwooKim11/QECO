from MEC_Env import MEC
from DDQN_torch_bilstm import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil
from sklearn.metrics import r2_score

'''
if not os.path.exists("models"):
    os.mkdir("models")
else:
    shutil.rmtree("models")
    os.mkdir("models")
'''
# 보상 값을 계산합니다. 작업이 완료되지 않았으면 패널티를 부과합니다. 그리고 공통으로 계산에 쓰인 에너지 소비를 차감합니다.
def reward_fun(ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy, delay, max_delay, unfinish_task):
    # edge_comp_energy 값중  e!=0 일때의 generator을 뽑아서 next에 넘겨줌. 그리고 next로 값 추출
    edge_energy  = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)
    penalty     = -max_delay*4
    if unfinish_task == 1:
        reward = penalty
    else:
        reward = 0
    reward = reward - (ue_comp_energy + ue_trans_energy + edge_energy + idle_energy)
    return reward

# 에피소드 t에서의 UE들의 평균 reward, delay, energy을 계신합니다.
def monitor_reward(ue_RL_list, episode):
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    print(f"reward: {avg_episode_sum_reward}")
    return avg_episode_sum_reward

# 에피소드 별로, ue의 task을 연산하는데 걸리는 시간
def monitor_delay(ue_RL_list, episode):
    delay_ue_list = [sum(ue_RL.delay_store[episode]) for ue_RL in ue_RL_list]
    avg_delay_in_episode = sum(delay_ue_list) / len(delay_ue_list)
    print(f"delay: {avg_delay_in_episode}")
    return avg_delay_in_episode

def monitor_energy(ue_RL_list, episode):
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    print(f"energy: {avg_energy_in_episode}")
    return avg_energy_in_episode

def cal_reward(ue_RL_list):
    total_sum_reward = 0
    num_episodes = 0
    for ue_num, ue_RL in enumerate(ue_RL_list):
        print("________________________")
        print("ue_num:", ue_num)
        print("________________________")
        for episode, reward in enumerate(ue_RL.reward_store):
            print("episode:", episode)
            reward_sum = sum(reward)
            print(reward_sum)
            total_sum_reward += reward_sum
            num_episodes += 1
    avg_reward = total_sum_reward / num_episodes
    print(total_sum_reward, avg_reward)



def train(ue_RL_list, NUM_EPISODE):
    global n_ue
    # """에피소드별로 평균 보상, 지연, 에너지 소비, 작업 드롭률을 추적하기 위한 리스트들을 초기화하고,
    #  강화 학습 스텝을 카운팅하는 변수 RL_step을 초기화합니다."""

    avg_reward_list = []
    avg_reward_list_2 = []
    avg_delay_list_in_episode = []
    avg_energy_list_in_episode = []
    num_task_drop_list_in_episode = []
    RL_step = 0
    a = 1



    for episode in range(NUM_EPISODE):
        print("episode  :", episode)
        print("epsilon  :", ue_RL_list[0].epsilon)
        # 현재 에피소드 번호와 탐험을 위한 epsilon 값(첫 번째 UE의 에이전트를 참조하여)을 출력합니다.


        # BITRATE ARRIVAL
        # 각 UE에 대한 작업 도착을 시뮬레이션합니다.
        # 이는 각 시간 단계에 대해 임의의 작업 크기와 도착 확률을 생성하고,
        # 최대 지연 시간을 고려하여 작업 배열을 조정합니다.

        bitarrive = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob  # TASK_ARRIVE_PROB: 작업 도착 확률 -> 30% 확률로 task 도착
        bitarrive = bitarrive * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        # delay가 발생하여서 뒤에 max_delay만큼은 not bit arrive -> 110-10 = 100 time_slot만 걸리게,, (근데 굳이?)
        bitarrive[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])
        print('bitarive:', bitarrive) # row 수 = 전체시간, col 수` = ue 개수

        # OBSERVATION MATRIX SETTING
        # 각 시간 단계와 UE에 대한 관찰, LSTM 상태, 행동,
        # 그리고 다음 상태를 추적하기 위한 history 리스트와 보상 지시자를 초기화합니다.
        # history는 experience의 일부 Algorithm1 Line 12
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        # 환경을 초기 상태로 리셋하고, 초기 관찰 및 LSTM 상태를 가져옵니다.
        observation_all, lstm_state_all = env.reset(bitarrive)

        # TRAIN DRL
        # 무한 루프를 시작하여 에이전트가 환경과 상호 작용하게 합니다.
        # 이 루프는 에피소드가 끝날 때(done == True일 때)까지 계속됩니다.

        while True:

            # PERFORM ACTION
            # 각 UE에 대해 행동을 결정하고 저장합니다.
            #  이는 현재 관찰에 기반하여 각 UE의 에이전트가 최적의 행동을 선택하게 합니다.

            # action_all -> 각 ue의 offloading decision Algorithm1 (line 4~7)
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :]) # -> 배열 차원을 간단히함. Ex) (1,2) -> (2,)
                if np.sum(observation) == 0:
                    # if there is no task, action = 0 (also need to be stored)
                    action_all[ue_index] = 0
                else:
                    # Algorithm1 line 7
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0: # 관측이 되었다면 action store. ques) obeservation(state)가 0 은 무슨 의미? 배열이잖아. -> task가 없다는 의미다.
                        # DQN에 메모리 저장
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            # 선택된 행동을 환경에 적용하고, 다음 상태, LSTM 상태, 그리고 에피소드가 끝났는지 여부를 반환받습니다.
            # Algorithm1 Line 9-10
            # action_all = (n_ue, )
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # should store this information in EACH time slot
            # 각 UE에 대한 정보를 업데이트하고, 환경에서 얻은 결과를 기반으로 학습 데이터(트랜지션)를 저장합니다.
            # LSTM 저장
            # Algorithm1 Line 11-12 XX
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index,:])

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            # observation_all, observation_all_ 차이 env.step 하기 전, 후의 값
            for ue_index in range(env.n_ue):
                obs = observation_all[ue_index, :]
                lstm = np.squeeze(lstm_state_all[ue_index, :])
                action = action_all[ue_index]
                obs_ = observation_all_[ue_index]
                lstm_ = np.squeeze(lstm_state_all_[ue_index,:])
                history[env.time_count - 1][ue_index].update({
                    'observation': obs,
                    'lstm': lstm,
                    'action': action,
                    'observation_': obs_,
                    'lstm_': lstm_
                })
                # reward_indicator 배열에서 특정 열의 값이 1이아니고, 해당요소의 env.process_delay가 0보다 큰 경우의 인덱스를 찾아 update_index에 저장
                update_index = np.where((1 - reward_indicator[:,ue_index]) *env.process_delay[:,ue_index] > 0)[0]
                # print(update_index)
                if len(update_index) != 0:
                    for time_index in update_index:
                        reward = reward_fun(
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index],
                            env.process_delay[time_index, ue_index],
                            env.max_delay,
                            env.unfinish_task[time_index, ue_index]
                        )
                        # Training Process Line 10
                        # transition(현재 state, action, reward, 다음 state)
                        ue_RL_list[ue_index].store_transition(
                            history[time_index][ue_index]['observation'],
                            history[time_index][ue_index]['lstm'],
                            history[time_index][ue_index]['action'],
                            reward,
                            history[time_index][ue_index]['observation_'],
                            history[time_index][ue_index]['lstm_']
                        )
                        ue_RL_list[ue_index].do_store_reward(
                            episode,
                            time_index,
                            reward
                        )
                        ue_RL_list[ue_index].do_store_delay(
                            episode,
                            time_index,
                            env.process_delay[time_index, ue_index]
                        )
                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy [time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index]
                        )
                        reward_indicator[time_index, ue_index] = 1

            # Algorithm2 Training Process Line 18
            # ADD STEP (one step does not mean one store)
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # Algorithm2 Training Process Line 19-20
            # CONTROL LEARNING START TIME AND FREQUENCY
            # 일정 스텝마다 모든 UE의 에이전트가 학습을 수행하도록 합니다.
            # 여기서는 200 스텝 이후부터 10의 배수 스텝마다 학습을 진행합니다.
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            # GAME ENDS
            # 에피소드가 끝났을 경우(예: 모든 작업이 처리되거나 시간이 끝났을 때),
            # 작업의 완료 상태, 드롭률, 에너지 소비 등을 계산하고 출력한 후 루프를 종료합니다.

            if done:
                for task in env.task_history:
                    cmpl = drp = 0
                    for t in task:
                        d_states = t['d_state']
                        if any(d < 0 for d in d_states):
                            t['state'] = 'D'
                            drp += 1
                        elif all(d > 0 for d in d_states):
                            t['state'] = 'C'
                            cmpl += 1
                full_complete_task = 0
                full_drop_task = 0
                complete_task = 0
                drop_task = 0
                for history in env.task_history:
                    for task in history:
                        if task['state'] == 'C':
                            full_complete_task += 1
                        elif task['state'] == 'D':
                            full_drop_task += 1
                        for component_state in task['d_state']:
                            if component_state == 1:
                                complete_task += 1
                            elif component_state == -1:
                                drop_task += 1
                # MEC line 170 self.task_history[ue_index].append(task_dic)
                # ?? (MD의 개수)*( MD 0의 history 개수)*n_compnent -> 왜 이렇게 했을까? 1. 모든 MD의 도착 task 수 같음 or 2. 대충 어림잡아 계산
                cnt = len(env.task_history) * len(env.task_history[0]) * env.n_component
                print(len(env.task_history), len(env.task_history[0]), cnt)
                print("++++++++++++++++++++++")
                print("drrop_rate   : ", full_drop_task/(cnt/env.n_component))
                print("full_drrop   : ", full_drop_task)
                print("full_complate: ", full_complete_task)
                print("complete_task: ", complete_task)
                print("drop_task:     ", drop_task)
                print("++++++++++++++++++++++")


                avg_reward_list.append(-(monitor_reward(ue_RL_list, episode)))
                if episode % 10 == 0:
                    avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode])/10)
                    avg_delay_list_in_episode.append(monitor_delay(ue_RL_list, episode))
                    avg_energy_list_in_episode.append(monitor_energy(ue_RL_list, episode))
                    print('avg delay:, energy')
                    print(avg_delay_list_in_episode, avg_energy_list_in_episode)
                    total_drop = full_drop_task
                    num_task_drop_list_in_episode.append(total_drop)


                    # Writing data to files

                    data = [avg_reward_list, avg_delay_list_in_episode, avg_energy_list_in_episode, num_task_drop_list_in_episode]
                    filenames = ['reward.txt', 'delay.txt', 'energy.txt', 'drop.txt']
                    for i in range(len(data)):
                        with open(filenames[i], 'w') as f:
                            f.write('\n'.join(str(x) for x in data[i]))
                # if episode == 0 or (episode+1)%50 == 0:
                #   drop_rate_list.append(full_drop_task/(cnt/env.n_component))
                #   drop_task_list.append(full_drop_task)
                #   complete_task_list.append(full_complete_task)

                # Process energy
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_comp_energy = sum(sum(env.ue_comp_energy))

                # Transmission energy
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                ue_tran_energy = sum(sum(env.ue_tran_energy))

                # edge energy
                edge_bit_processed = sum(sum(env.edge_bit_processed))
                edge_comp_energy = sum(sum(env.edge_comp_energy))
                ue_idle_energy = sum(sum(env.ue_idle_energy))

                # Print results
                print(int(ue_bit_processed), ue_comp_energy, "local")
                print(int(ue_bit_transmitted), ue_tran_energy, "trans")
                print(int(sum(edge_bit_processed)),sum(edge_comp_energy), sum(ue_idle_energy), "edge")
                print("_________________________________________________")

                break # Training Finished

    x=np.array(list(range(0, NUM_EPISODE//10)))
    x_r = np.array(list(range(0, NUM_EPISODE)))
    line_reward_1 = np.polyfit(x_r, avg_reward_list, 1)
    line_reward_2 = np.polyfit(x, avg_reward_list_2, 1)
    line_delay = np.polyfit(x, avg_delay_list_in_episode, 1)
    line_energy = np.polyfit(x, avg_energy_list_in_episode, 1)
    line_drop = np.polyfit(x, num_task_drop_list_in_episode, 1)
    x_minmax = np.array([0, NUM_EPISODE//10])
    x_r_minmax = np.array([0, NUM_EPISODE])

    fit_reward_y_1 = x_r_minmax * line_reward_1[0] + line_reward_1[1]
    fit_reward_y_2 = x_minmax * line_reward_2[0] + line_reward_2[1]
    fit_delay_y = x_minmax * line_delay[0] + line_delay[1]
    fit_energy_y = x_minmax * line_energy[0] + line_energy[1]
    fit_drop_y = x_minmax * line_drop[0] + line_drop[1]

    fig1, ax1 = plt.subplots(figsize=(8, 8))
    ax1.plot(avg_delay_list_in_episode, '-')
    ax1.set_ylabel('Average Delay(ms)')
    ax1.set_xlabel('epoch 수')
    ax1.plot(x_minmax, fit_delay_y, color='orange')
    fig1.savefig('fig-delay.png')

    # 두 번째 그래프를 위한 figure 생성
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    ax2.plot(avg_energy_list_in_episode, '-')
    ax2.set_ylabel('Average Energy(Jx10)')
    ax2.set_xlabel('epoch 수')
    ax2.plot(x_minmax, fit_energy_y, color='orange')
    fig2.savefig('fig-energy.png')

    est_reward_y_1 = x_r*line_reward_1[0] + line_reward_1[1]
    est_reward_y_2 = x*line_reward_2[0] + line_reward_2[1]
    est_delay_y = x*line_delay[0] + line_delay[1]
    est_energy_y = x*line_energy[0] + line_energy[1]
    est_drop_y = x*line_drop[0] + line_drop[1]

    r2s = [r2_score(avg_reward_list, est_reward_y_1),
            r2_score(avg_reward_list_2, est_reward_y_2),
            r2_score(avg_delay_list_in_episode, est_delay_y),
            r2_score(avg_energy_list_in_episode, est_energy_y),
            r2_score(num_task_drop_list_in_episode, est_drop_y)]

    print("R2 - Reward :", r2s[0])
    print("R2 - avg Reward :", r2s[1])
    print("R2 - Delay :", r2s[2])
    print("Delay: y = {0}x+{1}".format(line_delay[0], line_delay[1]))
    print("R2 - Energy :", r2s[3])
    print("Energy: y = {0}x+{1}".format(line_energy[0], line_energy[1]))
    print("R2 - Drop :", r2s[4])
    r2_file = 'R2-'+str(n_ue)+'.txt'
    with open(r2_file, 'w') as f:
      for i in range(len(r2s)):
        f.write('\n'.join(str(r2s[i])))


if __name__ == "__main__":

    # GENERATE ENVIRONMENT
    # n_ue = Config.N_UE
    n_ue = 50
    env = MEC(n_ue, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    ue_RL_list = list()
    for ue in range(n_ue):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate       = Config.LEARNING_RATE,
                                                    reward_decay        = Config.REWARD_DDECAY,
                                                    e_greedy            = Config.E_GREEDY,
                                                    replace_target_iter = Config.N_NETWORK_UPDATE,  # each 200 steps, update target net
                                                    memory_size         = Config.MEMORY_SIZE,  # maximum of memory
                                                    doLC                = False,  # Local Computing
                                                    doFO                = False, # Full Offloading
                                                    ))

    # LOAD MODEL
    '''
    for ue in range(Config.N_UE):
        ue_RL_list[ue].Initialize(ue_RL_list[ue].sess, ue)
    '''

    # TRAIN THE SYSTEM
    train(ue_RL_list, 500)



