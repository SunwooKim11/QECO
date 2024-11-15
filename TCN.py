import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


# TemporalBlock 클래스는 TCN의 기본적인 블록으로, dilated convolution을 사용하여 시계열 데이터의 긴 시점 의존성을 학습합니다.
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()

        # 첫 번째 합성곱 계층 (Conv2d) - (1, kernel_size) 커널 사용, dilation 적용 -
        # > 시계열 데이터는 1차원 데이터이기 때문에, 2d에서 한쪽 축 크기를 1로 하면 1차원 으로 됨.
        # weight_norm을 적용하여 학습 안정성 향상
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))

        # ZeroPadding - 시간 방향으로 padding 추가
        # padding=(padding, 0, 0, 0) 형식으로 좌측만 padding을 추가하여 causal 구조 보장
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))

        # ReLU 활성화 함수
        self.relu = nn.ReLU()

        # Dropout - 일부 뉴런을 무작위로 0으로 설정하여 과적합 방지
        self.dropout = nn.Dropout(dropout)

        # 두 번째 합성곱 계층 (Conv2d) - 첫 번째와 유사하지만 입력 채널과 출력 채널이 n_outputs로 설정
        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, (1, kernel_size),
                                           stride=stride, padding=0, dilation=dilation))

        # Sequential 네트워크 정의 - conv1, relu, dropout, conv2을 연속적으로 수행
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.dropout,
                                 self.pad, self.conv2, self.relu, self.dropout)

        # downsample: 입력과 출력의 차원이 다를 때, residual 연결을 위한 1x1 Conv1d 레이어 추가
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        # 추가 ReLU 활성화 함수
        self.relu = nn.ReLU()

        # 가중치 초기화
        self.init_weights()

    # 가중치 초기화 함수
    def init_weights(self):
        # 첫 번째 합성곱 계층 가중치를 정규분포로 초기화
        self.conv1.weight.data.normal_(0, 0.01)

        # 두 번째 합성곱 계층 가중치를 정규분포로 초기화
        self.conv2.weight.data.normal_(0, 0.01)

        # downsample 계층이 존재하는 경우, 가중치를 정규분포로 초기화
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    # 순전파 함수
    def forward(self, x):
        # 입력 텐서를 3차원으로 확장하여 Conv2d에 맞춤 -> self.net을 거친 후 다시 2차원으로 만듦
        out = self.net(x.unsqueeze(2)).squeeze(2)

        # Residual 연결 - downsample 계층이 있으면 사용하고, 없으면 입력을 그대로 사용
        res = x if self.downsample is None else self.downsample(x)

        # Residual과 out을 더한 후 ReLU를 적용하여 최종 출력
        return self.relu(out + res)


# TemporalConvNet 클래스는 TCN의 전체 네트워크를 구성합니다.
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()

        # 각 층을 담을 리스트 초기화
        layers = []

        # 네트워크의 레벨 수 결정 (num_channels 리스트의 길이)
        num_levels = len(num_channels)

        # 각 레벨에 대해 TemporalBlock을 생성하여 layers에 추가
        for i in range(num_levels):
            # dilation 값은 2의 거듭제곱으로 증가 (1, 2, 4, ...)
            dilation_size = 2 ** i
            # 첫 번째 레벨은 입력 채널이 num_inputs, 이후 레벨은 이전 레벨의 출력 채널
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            # 현재 레벨의 출력 채널
            out_channels = num_channels[i]

            # TemporalBlock 생성 후 layers 리스트에 추가
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        # Sequential 네트워크로 구성하여 network 속성에 저장
        self.network = nn.Sequential(*layers)

    # 순전파 함수 - 입력 x를 network에 전달
    def forward(self, x):
        return self.network(x)
