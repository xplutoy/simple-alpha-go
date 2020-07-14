# -*- coding: utf-8 -*-
# @Time  : 2019/3/1 11:05
# @Author : yx
# @Desc : ==============================================

from config import *
from gobang import Board
from utils import *


class Node(object):
    def __init__(self, parent=None, index=0, level=0):
        self.index = index  # MCTS中节点索引，方便画图显示
        self.level = level  # 树的层级
        self.parent = parent
        self.edges = []  # [(act, na, wa, qa, pa)]
        self.child = []  # [node]

    def is_leaf(self):
        return len(self.edges) == 0

    def _zip(self):
        assert len(self.edges) != 0
        return list(zip(*self.edges))

    def pucts(self):
        assert not self.is_leaf()
        _, nv, _, qv, pv = self._zip()
        return np.array(qv) + C_PUCT * np.array(pv) * np.sqrt(sum(nv)) / (1 + np.array(nv))

    def expand(self, valid_acts, probs):
        assert self.is_leaf()
        for a in valid_acts:
            self.edges.append((a, 0, 0, 0, probs[a]))
            self.child.append(Node(self, self.index, self.level + 1))
            self.index += 1

    def select(self):
        idx = int(np.argmax(self.pucts()))
        return idx, self.edges[idx][0], self.child[idx]

    def update_edge_stats(self, act_idx, v):
        act, na, wa, qa, pa = self.edges[act_idx]
        na += 1
        wa += v
        qa = wa / na
        self.edges[act_idx] = (act, na, wa, qa, pa)

    def valid_act_probs(self, tau):
        av, nv, _, _, _ = self._zip()
        return av, softmax(1.0 / tau * np.log(np.array(nv) + 1e-10))


class Mcts:
    def __init__(self, net, borad=None):
        self.index = 0  # 添加节点索引
        self.root = Node()
        if borad is None:
            self.board = Board()
        else:
            self.board = borad
        self.policy_value_net = net

    def _play_out(self):
        """
        :param board:
        :return: leaf, v
        """
        leaf = self.root
        board = self.board.clone()
        act_idx_lst = []
        while not leaf.is_leaf():
            idx, act, leaf = leaf.select()
            act_idx_lst.append(idx)
            row, col = board.action_1_to_2(act)
            board.move(row, col)

        res = board.judge()
        if res == 0:  # 棋局未结束
            valid_acts = board.valid_acts()
            pv, v = self.policy_value_net.get_policy_value(board.current_state())
            pv, v = np.squeeze(pv), float(np.squeeze(v))
            leaf.expand(valid_acts, pv)
        elif res == 2:  # 平局
            v = 0
        else:  # 黑胜 或者 白胜(这里胜利针对的是当前节点的父节点，所以v取负)
            v = -1

        return leaf, act_idx_lst, v

    def backup(self, leaf, act_idx_lst, v):
        # 从叶子节点开始回传(不包括新增的扩展节点)
        v = -v
        if leaf.parent is not None:
            leaf.parent.update_edge_stats(act_idx_lst.pop(), v)
            self.backup(leaf.parent, act_idx_lst, v)

    def simulate(self, n):
        for _ in range(n):
            leaf, act_idx_lst, v = self._play_out()
            self.backup(leaf, act_idx_lst, v)

    def play(self, tau, with_noise=False):
        # 构造pi
        pi = np.zeros(self.board.board_size ** 2)
        av, probs = self.root.valid_act_probs(tau)
        pi[list(av)] = probs

        # 选取act时是否要添加额外的噪声
        if with_noise:
            probs = (1 - NOISE_EPS) * probs + NOISE_EPS * np.random.dirichlet(ALPHA * np.ones(len(probs)))
        idx = np.random.choice(len(probs), p=probs)
        act = self.root.edges[idx][0]

        return pi, act, idx

    def shift_root(self, act, idx):
        # root 节点下移, 棋盘状态变更
        self.root = self.root.child[idx]
        self.root.parent = None
        row, col = self.board.action_1_to_2(act)
        self.board.move(row, col)
