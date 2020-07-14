#!/usr/bin/python3
# _*_ coding:utf-8 _*_
# Author: Xue Mingfeng

import threading
import tkinter as tk
from tkinter.simpledialog import messagebox

import numpy as np

from config import BOARD_SIZE, CONNECT_N

GRID_SIZE = 50  # 每个格子的大小都是50x50
PIECE_SIZE = 20  # 棋子的半径


class Board:
    def __init__(self, board_size=BOARD_SIZE, connect_n=CONNECT_N):
        self.board_size = board_size
        self.connect_n = connect_n
        self.board = np.zeros(shape=[board_size, board_size], dtype=np.int8)
        # 默认黑子先行， -1：黑子的轮次， 1： 白子的轮次, 同时黑色棋子也为:-1,白色棋子：1
        self.color = -1
        self.turn_num = 0  # 已经进行的轮数
        self.last_move_position = None  # 上一手落子的位置

    def clone(self):
        board = Board(self.board_size, self.connect_n)
        board.color = self.color
        board.board = self.board.copy()
        board.turn_num = self.turn_num
        board.last_move_position = self.last_move_position
        return board

    def move(self, row, col):
        assert self.board[row][col] == 0

        self.board[row][col] = self.color
        self.last_move_position = [row, col]
        self.color = -self.color
        self.turn_num += 1

    def current_state(self):
        X, Y = self.board.copy(), self.board.copy()
        if self.color == -1:
            X = -X
            X[X < 0] = 0
            Y[Y < 0] = 0
            F = np.ones([BOARD_SIZE, BOARD_SIZE])
        else:
            X[X < 0] = 0
            Y = -Y
            Y[Y < 0] = 0
            F = np.zeros([BOARD_SIZE, BOARD_SIZE])
        L = np.zeros([BOARD_SIZE, BOARD_SIZE])
        if self.last_move_position is not None:
            L[self.last_move_position[0], self.last_move_position[1]] = -self.color
        state = np.stack([X, F, Y, L], 0)  # shape 4 x board_size x board_size
        return state

    def judge(self):
        if self.last_move_position is None:
            # 棋局刚开始
            return 0

        # 0表示黑子，1 表示空，2表示白子
        board = self.board + 1
        win_str = '2' * self.connect_n if self.color == -1 else '0' * self.connect_n

        temp_row = self.last_move_position[0]
        temp_col = self.last_move_position[1]
        row = temp_row
        col = temp_col
        # 判断落子从左上到右下的斜边是否5连
        while row > 0 and col > 0:
            row = row - 1
            col = col - 1
        line = ''
        while row < BOARD_SIZE and col < BOARD_SIZE:
            line += str(board[row, col])
            row = row + 1
            col = col + 1
        if win_str in line:
            return -self.color

        # 判断从落子右上到左下是否5连
        row = temp_row
        col = temp_col
        while row > 0 and col < BOARD_SIZE - 1:
            row = row - 1
            col = col + 1
        line = ''
        while row < BOARD_SIZE and col > -1:
            line += str(board[row, col])
            row = row + 1
            col = col - 1
        if win_str in line:
            return -self.color
        # 判断落子竖直方向是否5连
        row = temp_row
        col = temp_col
        while row > 0:
            row = row - 1
        line = ''
        while row < BOARD_SIZE:
            line += str(board[row, col])
            row += 1
        if win_str in line:
            return -self.color
        # 判断落子水平方向是否5连
        row = temp_row
        col = temp_col
        while col > 0:
            col = col - 1
        line = ''
        while col < BOARD_SIZE:
            line += str(board[row, col])
            col += 1
        if win_str in line:
            return -self.color

        # 判断是否平局
        if self.turn_num >= self.board_size ** 2:
            return 2

        # 棋局未结束
        return 0

    def action_1_to_2(self, action):
        row = int(action / self.board_size)
        col = action - row * self.board_size
        return row, col

    def action_2_to_1(self, row, col):
        return row * self.board_size + col

    def valid_acts(self):
        act = []
        board_flatten = self.board.flatten()
        for i in range(len(board_flatten)):
            if board_flatten[i] == 0:
                act.append(i)
        return act


class Gobang:
    def __init__(self, web_server=False, with_ai=False, ckpt=None):
        self.board = Board()
        self.with_ai = with_ai
        if self.with_ai:
            if not self.init_ai(ckpt):
                return None
        self.init_gui(web_server)

    def init_gui(self, web_server):
        if web_server:
            return 0
            # TODO
        else:
            self.window = tk.Tk()
            self.window.resizable(False, False)
            self.window.title('gobang')
            height = 50 * self.board.board_size
            width = height + 100
            self.window.geometry(str(width) + 'x' + str(height))
            self.canvas = tk.Canvas(self.window, bg='white', height=height, width=height)
            self.canvas.place(x=0, y=0)
            self.canvas.bind('<Button-1>', self.click_move)
            btn_reset = tk.Button(self.window, text='reset', command=self.load)
            btn_reset.place(x=height + 10, y=30, height=28, width=80)
            btn_black = tk.Button(self.window, text='黑子执子', command=self.use_black)
            btn_black.place(x=height + 10, y=60, height=28, width=80)
            btn_white = tk.Button(self.window, text='白子执子', command=self.use_white)
            btn_white.place(x=height + 10, y=90, height=28, width=80)
            btn_ai_first = tk.Button(self.window, text='AI执子', command=self.ai_move)
            btn_ai_first.place(x=height + 10, y=120, height=28, width=80)
            self.render_board()

    def init_ai(self, ckpt):
        from alpha_zero import AlphaZeroAgent
        if ckpt is not None:
            self.alpha_zero = AlphaZeroAgent().load(ckpt)
            return True
        else:
            print('请传入AI所使用的ckpt')
            return False

    def click_move(self, event):
        x = event.x
        y = event.y
        row = 0
        col = 0
        if x < self.board.board_size * 50 and y < self.board.board_size * 50:
            for i in range(self.board.board_size):
                if x >= i * GRID_SIZE and x <= (i + 1) * GRID_SIZE:
                    col = i
            for i in range(self.board.board_size):
                if y >= i * GRID_SIZE and y <= (i + 1) * GRID_SIZE:
                    row = i
            if self.board.board[row][col] == 0:
                self.board.move(row, col)
                result = self.board.judge()
                if result == 0:
                    if self.with_ai:
                        self.ai_move()
                else:
                    if result == -1:
                        messagebox.showinfo('游戏结果', '游戏结束，黑子胜')
                    if result == 1:
                        messagebox.showinfo('游戏结果', '游戏结束，白子胜')
                    if result == 2:
                        messagebox.showinfo('游戏结果', '游戏结束，平局')
                    self.load()

    def ai_move(self):
        act = self.alpha_zero.free_play(self.board, self.alpha_zero.agent)
        act = self.board.action_1_to_2(act)
        self.board.move(act[0], act[1])
        result = self.board.judge()
        if result == -1:
            messagebox.showinfo('游戏结果', '游戏结束，黑子胜')
        if result == 1:
            messagebox.showinfo('游戏结果', '游戏结束，白子胜')
        if result == 2:
            messagebox.showinfo('游戏结果', '游戏结束，平局')
        if result != 0:
            self.load()

    def use_black(self):
        self.board.color = -1

    def use_white(self):
        self.board.color = 1

    def load(self, board=None):
        if board is None:
            self.board = Board()
        else:
            self.board = board

    def render_board(self):
        '''
        显示棋盘
        '''
        self.timer = threading.Timer(0.01, self.render)
        self.timer.setDaemon(True)
        self.timer.start()
        # 画线
        for i in range(self.board.board_size):
            line = self.canvas.create_line(25 + i * GRID_SIZE, 25, 25 + i * GRID_SIZE,
                                           self.board.board_size * 50 - 25,
                                           fill='gray')
        for i in range(self.board.board_size):
            line = self.canvas.create_line(25, 25 + i * GRID_SIZE, self.board.board_size * 50 - 25,
                                           25 + i * GRID_SIZE,
                                           fill='gray')
        # 画星和天元
        if self.board.board_size == 15:
            self.canvas.create_oval(25 + 3 * GRID_SIZE - 3, 25 + 3 * GRID_SIZE - 3, 25 + 3 * GRID_SIZE + 3,
                                    25 + 3 * GRID_SIZE + 3, fill='black')
            self.canvas.create_oval(25 + 3 * GRID_SIZE - 3, 25 + 11 * GRID_SIZE - 3, 25 + 3 * GRID_SIZE + 3,
                                    25 + 11 * GRID_SIZE + 3, fill='black')
            self.canvas.create_oval(25 + 11 * GRID_SIZE - 3, 25 + 3 * GRID_SIZE - 3, 25 + 11 * GRID_SIZE + 3,
                                    25 + 3 * GRID_SIZE + 3, fill='black')
            self.canvas.create_oval(25 + 11 * GRID_SIZE - 3, 25 + 11 * GRID_SIZE - 3, 25 + 11 * GRID_SIZE + 3,
                                    25 + 11 * GRID_SIZE + 3, fill='black')
            self.canvas.create_oval(25 + 7 * GRID_SIZE - 3, 25 + 7 * GRID_SIZE - 3, 25 + 7 * GRID_SIZE + 3,
                                    25 + 7 * GRID_SIZE + 3, fill='black')
        self.window.mainloop()

    # 本函数不直接调用，只使用在计时器中用于更新棋子状态
    def render(self):
        for row in range(self.board.board_size):
            for col in range(self.board.board_size):
                if self.board.board[row, col] == -1:
                    self.canvas.create_oval(25 + col * GRID_SIZE - PIECE_SIZE, 25 + row * GRID_SIZE - PIECE_SIZE,
                                            25 + col * GRID_SIZE + PIECE_SIZE, 25 + row * GRID_SIZE + PIECE_SIZE,
                                            fill='black', tags=('chess', str(row) + '_' + str(col)))
                if self.board.board[row, col] == 1:
                    self.canvas.create_oval(25 + col * GRID_SIZE - PIECE_SIZE, 25 + row * GRID_SIZE - PIECE_SIZE,
                                            25 + col * GRID_SIZE + PIECE_SIZE, 25 + row * GRID_SIZE + PIECE_SIZE,
                                            fill='white', tags=('chess', str(row) + '_' + str(col)))
                if self.board.board[row, col] == 0:
                    self.canvas.delete(str(row) + '_' + str(col))
        global timer
        timer = threading.Timer(0.01, self.render)
        timer.start()


if __name__ == '__main__':
    showboard = Gobang(with_ai=True, ckpt='pytorch_6_6_4.model')
