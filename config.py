# MSCT
TAU = 1
TAU_INFINITE = 1e-3
C_PUCT = 5  # A higher value means relying on the prior more.
ALPHA = 0.55  # Dirichlet noise的参数
NOISE_EPS = 0.25

SELF_PLAY_SIMULATION_NUM = 400
SELF_PLAY_GAME_NUM = 2  # self-play的游戏次数
FREE_PLAY_SIMULATION_NUM = 400
REPLAY_BUFF_CAPACITY = 50000  # replay_buffer的容量,论文中是存储最近的500000把游戏的数据

# 训练参数
LR = 2e-3  # 学习率
BATCH_SIZE = 256
PIPELINE_ITER_NUM = 2000  # alpha_zero pipeline迭代次数
L2_WEIGHT_DECAY = 1e-4  # l2_weight_decay 参数
TRAIN_BATCH_NUM = 5  # 网络训练迭代次数
FREQUENCY_TO_SAVE = 10
EVAL_GAME_NUM = FREQUENCY_TO_SAVE * SELF_PLAY_GAME_NUM  # 模型之间评估所用的游戏次数

# 网络其他
FILTER_NUM = 256  # 卷积核的数量
RES_BLOCK_NUM = 3
SAVE_MODEL_DIR = './ckpts6x6/'

# Env
BOARD_SIZE = 6  # 棋盘大小
CONNECT_N = 4  # 达到几连游戏结束

# 其他
USE_PYTORCH = True
# USE_PYTORCH = False
