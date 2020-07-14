# -*- coding: utf-8 -*-
# @Time  : 2019/3/8 12:38
# @Author : yx
# @Desc : ==============================================
import multiprocessing as mp
import random
from collections import deque

from logger import logger
from mcts import *

if USE_PYTORCH:
    from policy_value_net_pytorch import PolicyValueNet as PVNet
else:
    from policy_value_net_tf import PolicyValueNet as PVNet


# 根据五子棋棋盘的对称性增广数据
def _augment_play_data(data):
    """
    :param data: [(state, pi, z)]  state: [3xNxN]
    :return:
    """
    extend_data = []
    for state, pi, z in data:
        pi = pi.reshape(BOARD_SIZE, BOARD_SIZE)
        for i in [1, 2, 3, 4]:
            for j in [True, False]:
                ss = np.array([np.rot90(s, i) for s in state])
                pp = np.rot90(pi, i)
                if j:
                    ss = np.array([np.fliplr(s) for s in ss])
                    pp = np.fliplr(pp)
                extend_data.append((ss, pp.ravel(), z))
    return extend_data


class AlphaZeroAgent:
    def __init__(self, lr_factor=1, kl_datum=0.02):
        self.agent = PVNet(BOARD_SIZE)
        self.reply_buffer = deque(maxlen=REPLAY_BUFF_CAPACITY)
        self.lr_factor = lr_factor
        self.kl_datum = kl_datum

    def load(self, model_path):
        self.agent.restore_model(model_path)
        return self

    def self_play(self, n):
        res_lst, step_lst = [], []
        s_lst, c_lst, p_lst, z_lst = [], [], [], []
        for _ in range(n):
            step, res = 0, 0  # 下棋的轮数和结果
            mcts = Mcts(self.agent)
            while res == 0:  # 非终局
                mcts.simulate(SELF_PLAY_SIMULATION_NUM - 10)
                s_lst.append(mcts.board.current_state())
                c_lst.append(mcts.board.color)
                tau = TAU if step < 20 else TAU_INFINITE
                pi, act, idx = mcts.play(tau, with_noise=True)
                p_lst.append(pi)
                mcts.shift_root(act, idx)
                res = mcts.board.judge()
                step += 1

            res_lst.append(res)
            step_lst.append(step)
            # 棋盘局势  -1黑胜，0对局未完，1白胜，2平局 跟环境保持一致
            if res == 2:  # 平局
                z = 0
            elif res == -1:  # 黑棋胜
                z = -1
            else:  # 白棋胜
                z = 1
            z_lst.extend([z * color for color in c_lst])
        # logging
        logger.info('self_play(%d局) 黑vs白：[%d/%d] 局长：%s' % (n, res_lst.count(-1), res_lst.count(1), str(step_lst)))
        return zip(s_lst, p_lst, z_lst)

    @staticmethod
    def free_play(bord, agent):
        mcts = Mcts(agent, bord)
        mcts.simulate(FREE_PLAY_SIMULATION_NUM)
        _, act, _ = mcts.play(TAU_INFINITE)
        return act

    @staticmethod
    def eval(conn):
        """
        :param conn:
        :return:
        """

        players = [PVNet(BOARD_SIZE, device_str='cuda:1'), PVNet(BOARD_SIZE, device_str='cuda:1')]
        while True:
            scores = [0, 0]
            model_paths = conn.recv()
            if model_paths is None:
                conn.close()
                break
            players[0].restore_model(model_paths[0])
            players[1].restore_model(model_paths[1])
            for i in range(EVAL_GAME_NUM):
                idx, _ = AlphaZeroAgent.duiyi(Board(), players, i % 2)
                if idx is not None:
                    scores[idx] += 1
            conn.send(scores)

    @staticmethod
    def duiyi(chess_board, players, pidx=0):
        """ 对弈
        :param chess_board: 初始棋盘
        :param players: 双方棋手
        :param pidx: 先手的索引: 0 or 1, 默认第一个棋手先走
        :return: 平局：None 否则返回胜者的索引
        """
        assert len(players) == 2

        act_lst = []
        board = chess_board.clone()
        res = board.judge()
        while res == 0:
            act = AlphaZeroAgent.free_play(board, players[pidx % 2])
            act_lst.append(act)
            pidx += 1
            row, col = board.action_1_to_2(act)
            board.move(row, col)
            res = board.judge()

        if res == 2:  # 平局
            winner = None
        else:
            winner = (pidx + 1) % 2

        return winner, act_lst

    def self_play_training_pipeline(self):
        # self-play
        play_data_lst = _augment_play_data(self.self_play(SELF_PLAY_GAME_NUM))
        random.shuffle(play_data_lst)
        self.reply_buffer.extend(play_data_lst)
        if len(self.reply_buffer) < BATCH_SIZE:
            return

        # optimize
        for _ in range(TRAIN_BATCH_NUM):
            batch = random.sample(self.reply_buffer, BATCH_SIZE)
            states, probs, zs = zip(*batch)
            loss, kl, entropy = self.agent.train_step(states, probs, zs, self.lr_factor * LR)

            # adaptively adjust the learning rate
            if kl > self.kl_datum * 1.5 and self.lr_factor > 0.01:
                self.lr_factor /= 1.5
            elif kl < self.kl_datum / 2 and self.lr_factor < 100:
                self.lr_factor *= 1.5

            logger.info('loss: %.3f kl: %.3f entropy: %.3f lr_factor: %.3f' % (loss, kl, entropy, self.lr_factor))

        # slow window trick
        if len(self.reply_buffer) < self.reply_buffer.maxlen:
            [self.reply_buffer.popleft() for _ in range(2 * 8)]


if __name__ == '__main__':
    import os
    import datetime

    os.makedirs(SAVE_MODEL_DIR, exist_ok=True)
    mp.set_start_method('spawn')

    to_job, to_self = mp.Pipe()
    alg = AlphaZeroAgent()
    eval_p = mp.Process(target=alg.eval, args=(to_self,))
    eval_p.start()

    best_model = alg.agent.save_model(SAVE_MODEL_DIR + 'alpha_gobang' + '_0')
    # train
    model_paths = None
    for it in range(PIPELINE_ITER_NUM):
        logger.info('%s [AlphaZero Training] [迭代：%d]' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), it))

        if it % FREQUENCY_TO_SAVE == 0:
            save_model_path = SAVE_MODEL_DIR + 'alpha_gobang' + '_' + str(it)
            alg.agent.save_model(save_model_path)
            model_paths = [best_model, save_model_path]
            to_job.send(model_paths)

        alg.self_play_training_pipeline()

        if (it + 1) % FREQUENCY_TO_SAVE == 0:
            scores = to_job.recv()
            best_model = model_paths[0] if scores[0] > scores[1] else model_paths[1]
            logger.info(
                '当前最好模型：[%s] 评估结果：[%s : %s] [%d/%d]' %
                (best_model, model_paths[0], model_paths[1], scores[0], scores[1]))
    to_job.send(None)
    eval_p.join()
