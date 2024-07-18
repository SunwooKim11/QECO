

class Config(object):
    # 1. 시뮬레이션 환경 관련 설정

    # N_UE: 사용자 장비(UE)의 수
    # N_EDGE: 엣지 서버의 수
    # N_COMPONENT: 컴포넌트의 수 ?
    # N_EPISODE: 에피소드의 수
    # N_TIME_SLOT: 시간 슬롯의 수

    # MAX_DELAY: 최대 딜레이 ??
    # N_TIME: 총 시간 ??

    N_UE        = 50
    N_EDGE      = 5

    N_COMPONENT = 1

    N_EPISODE   = 1000

    N_TIME_SLOT = 100     
    MAX_DELAY   = 10
    N_TIME      = N_TIME_SLOT + MAX_DELAY

    # 2. 에너지 소모 관련 설정 -> 17번 수식

    # UE_COMP_ENERGY: UE의 연산 에너지
    # UE_TRAN_ENERGY: UE의 전송 에너지
    # UE_IDLE_ENERGY: UE의 대기 에너지
    # EDGE_COMP_ENERGY: 엣지 서버의 연산 에너지

    UE_COMP_ENERGY   = 2 # 2
    UE_TRAN_ENERGY   = 2.3 # 2.3
    UE_IDLE_ENERGY   = 0.1
    EDGE_COMP_ENERGY = 5  #5

    # 3. 시뮬레이션 시간 관련 설정

    # DURATION: 한 시간 슬롯의 지속 시간

    # 4. 용량(capacity) 관련 설정

    # UE_COMP_CAP: UE의 연산 용량
    # UE_TRAN_CAP: UE의 전송 용량
    # EDGE_COMP_CAP: 엣지 서버의 연산 용량

    DURATION         = 0.1
    UE_COMP_CAP      = 2.5
    UE_TRAN_CAP      = 14
    EDGE_COMP_CAP    = 41.8

    # 5. 작업(task) 관련 설정

    # TASK_COMP_DENS: 작업의 연산 밀도
    # TASK_ARRIVE_PROB: 작업 도착 확률
    # TASK_MIN_SIZE: 작업의 최소 크기
    # TASK_MAX_SIZE: 작업의 최대 크기

    TASK_COMP_DENS   = 0.297
    TASK_ARRIVE_PROB = 0.3
    TASK_MIN_SIZE    = 2
    TASK_MAX_SIZE    = 5

    # 6. 학습 관련 설정

    # LEARNING_RATE: 학습률
    # E_GREEDY: 엡실론 그리디

    # REWARD_DDECAY: 보상 감소율 ?
    # N_NETWORK_UPDATE: 네트워크 업데이트 수 ?
    # MEMORY_SIZE: 메모리 크기 ?

    LEARNING_RATE    = 0.01
    E_GREEDY         = 0.99
    REWARD_DDECAY = 0.9
    N_NETWORK_UPDATE = 200
    MEMORY_SIZE      = 500


