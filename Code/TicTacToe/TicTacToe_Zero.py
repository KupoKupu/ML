# importing Packages from tkinter
import sys
import os
import tkinter
from tkinter import *
from tkinter import messagebox
import torch
from torch import nn
from torch import distributions
import torch.nn.functional as F
import math
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
from torch.distributions import Categorical



if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = 'cpu'

stop_game = False
train = False

size = 6
win_size = 4

b = [[0] * size for i in range(size)]

states = torch.zeros(size, size).to(device)

Traj = []

class action_V(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(1, 64, 4)
        self.norm1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.norm2 = nn.BatchNorm2d(1)
        self.norm3 = nn.BatchNorm1d(1)
        self.norm4 = nn.BatchNorm1d(1)
        self.norm5 = nn.BatchNorm1d(1)
        self.norm3v = nn.BatchNorm1d(1)
        self.norm4v = nn.BatchNorm1d(1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, size ** 2)
        self.relu = nn.LeakyReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=0)
        self.fcv1 = nn.Linear(256, 256)
        self.fcv2 = nn.Linear(256, 256)
        self.fcv3 = nn.Linear(256, 1)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = self.conv1(x.unsqueeze(0)).view(64, 1, size - 3, size - 3)
        x = self.relu(self.norm1(x)).view(1, 64, size - 3, size - 3)
        x = self.conv2(x).view(256, 1, size - 5, size - 5)
        x = self.relu(self.norm2(x)).view(256)
        x1 = self.fc1(x).view(256, 1)
        x1 = self.relu(self.norm3(x1)).view(256)
        x1 = self.fc2(x1).view(256, 1)
        x1 = self.relu(self.norm4(x1)).view(256)
        x1 = self.fc3(x1).view(-1, 1)
        x1 = self.norm5(x1).view(-1)
        x1 = self.LogSoftmax(x1)
        v = self.fcv1(x).view(256, 1)
        v = self.relu(self.norm3v(v)).view(256)
        v = self.fcv2(v).view(256, 1)
        v = self.relu(self.norm4v(v)).view(256)
        v = self.tanh(self.fcv3(v))
        return x1, v

Kupo = action_V(size).to(device)
optimizer = torch.optim.Adam(Kupo.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))

class MCTS:
    def __init__(self, s, eta, n, player):
        self.states = s.clone()
        self.Q = torch.zeros(size ** 2).to(device)
        self.N = torch.zeros(size ** 2).to(device)
        self.eta = eta
        self.stop = False
        self.times = n
        self.p = player

    def check_if_win(self, s, player):
        if check_consecutive(s, win_size, player) > 0:
            self.stop = True

    def check_tie(self, s):
        if all([all(row) for row in s]):
            self.stop = True

    def fast_rollout(self):
        s = self.states.clone()
        score = (self.Q + self.eta * torch.exp(Kupo(s)[0])) / (1 + self.N)
        legal_mask = (s != 0).view(-1).to(device)
        a = torch.argmax(torch.Tensor.masked_fill(score, legal_mask, -torch.inf))
        self.N[a] += 1
        m, n = a // size, a % size
        s[m][n] = self.p
        self.check_if_win(s, self.p)
        if self.stop:
            self.Q[a] += (Kupo(s)[1].item() + 1) / 2
        self.check_tie(s)
        if self.stop:
            self.Q[a] += Kupo(s)[1].item() / 2
        if not self.stop:
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(torch.exp(Kupo(s)[0]) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = 3 - self.p
            value = Kupo(s)[1].item()
            self.check_if_win(s, 3 - self.p)
            if self.stop:
                self.Q[a] += (value - 1) / 2
            self.check_tie(s)
            if self.stop:
                self.Q[a] += value / 2
        while not self.stop:
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(torch.exp(Kupo(s)[0]) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = self.p
            self.check_if_win(s, self.p)
            if self.stop:
                self.Q[a] += (value + 1) / 2
                break
            self.check_tie(s)
            if self.stop:
                self.Q[a] += value / 2
                break
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(torch.exp(Kupo(s)[0]) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = 3 - self.p
            self.check_if_win(s, 3 - self.p)
            if self.stop:
                self.Q[a] += (value - 1) / 2
                break
            self.check_tie(s)
            if self.stop:
                self.Q[a] += value / 2
                break
        self.stop = False

    def output(self):
        for i in range(self.times):
            self.fast_rollout()
        return self.N


def p1_move(r, c):
    global states
    global b

    b[r][c].configure(text="X")
    states[r][c] = 1


def p_move(player):
    global states
    global b
    global Traj

    current_state = states.clone().to(device)

    mct = MCTS(states, 0.5, 150, player)
    N = mct.output()
    # print("P:", player, N)
    idx = torch.argmax(N)
    m, n = idx // size, idx % size
    if not train:
        b[m][n].configure(text="O")
    states[m][n] = player
    current_traj = [current_state, N / N.sum()]
    Traj.append(current_traj)

def is_all_equal(lst):
    return  lst[0] != 0 and all(x == lst[0] for x in lst)

def check_consecutive(s, n, player):
    j_list = range(size - n + 1)
    counter = 0

    for i in range(size):
        for j in j_list:
                if s[i][j] == player and is_all_equal([s[i][k] for k in range(j, n + j)]):
                    counter += 1
                if s[j][i] == player and is_all_equal([s[k][i] for k in range(j, n + j)]):
                    counter += 1

    for i in j_list:
        for j in j_list:
            if s[i][j] == player and is_all_equal([s[i + k][j + k] for k in range(n)]):
                counter += 1

    for i in j_list:
        for j in range(n - 1, size):
            if s[i][j] == player and is_all_equal([s[i + k][j - k] for k in range(n)]):
                counter += 1

    return counter

def check_if_win(player):
    global stop_game
    if check_consecutive(states, win_size, player) > 0:
        stop_game = True

def check_tie():
    global stop_game
    if not stop_game and all([all(row) for row in states]):
        stop_game = True

def update(ending):
    global Traj
    global states
    if not train:
        if ending == 3:
            messagebox.showinfo("tie", "Tie")
        else:
            messagebox.showinfo("Winner", f"Player {ending}" + " Won")
    else:
        action_v_update(ending)
        if ending == 1:
            print("P1 won!")
        elif ending == 2:
            print("P2 won!")
        elif ending == 3:
            print('Tie!')

def action_v_update(ending):
    global Traj
    score = 0
    if ending == 1:
        score = -1
    if ending == 2:
        score = 1
    optimizer.zero_grad()
    v_loss = 0
    a_loss = 0
    for [s, n] in Traj[0::2]:
        a_loss += nn.CrossEntropyLoss()(Kupo(s)[0], n)
        v_loss += (Kupo(s)[1] - score) ** 2
    for [s, n] in Traj[1::2]:
        a_loss += nn.CrossEntropyLoss()(Kupo(s)[0], n)
        v_loss += (Kupo(s)[1] + score) ** 2
    loss = v_loss + a_loss
    print("loss:", loss.item())
    loss.backward()
    optimizer.step()


def load():
    file_path = Path('64checkpoint.pth')
    if file_path.is_file():
        cp = torch.load('64checkpoint.pth')
        optimizer.load_state_dict(cp['64optimizer_state_dict'])
        Kupo.load_state_dict(cp['64model_state_dict'])


def save():
    cp = {
        '64model_state_dict': Kupo.state_dict(),
        '64optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(cp, '64checkpoint.pth')


def train_loop(rounds):
    global stop_game
    global Traj
    global states
    global train

    Kupo.train()
    train = True

    p1_win = 0
    p2_win = 0

    load()

    for i in range(rounds):
        print('Round ' + str(i + 1))
        stop_game = False
        states = torch.zeros(size, size).to(device)
        Traj = []

        p_move(2)

        while not stop_game:
            check_tie()
            if stop_game:
                update(3)
                break

            p_move(1)
            check_if_win(1)
            if stop_game:
                update(1)
                p1_win += 1
            else:
                check_tie()
                if stop_game:
                    update(3)
                    break
                p_move(2)
                check_if_win(2)
                if stop_game:
                    update(2)
                    p2_win += 1

    save()

    print('P1 won', p1_win, 'out of', rounds, 'rounds' )
    print('P2 won', p2_win, 'out of', rounds, 'rounds')


def move(r, c):
    global Traj

    if states[r][c] == 0:
        if not stop_game:
            p1_move(r, c)
            check_if_win(1)
            if stop_game:
                update(1)
            else:
                check_tie()
                if stop_game:
                    update(3)

        if not stop_game:
            p_move(2)
            check_if_win(2)
            if stop_game:
                update(2)
            else:
                check_tie()
                if stop_game:
                    update(3)




root = Tk()
root.title("Kupo\'s TicTacToe")

def new_game():
    global train
    train = False
    Kupo.eval()

    for i in range(size):
        for j in range(size):
            b[i][j] = Button(
                height=24 // size, width=48 // size + 1,
                font=("Helvetica", "20"),
                command=lambda r=i, c=j: move(r, c)
                )
            b[i][j].grid(row=i, column=j)

    p_move(2)

def main(rounds, tr):
    if tr == 1:
        train_loop(rounds)
    else:
        load()
        new_game()
        root.mainloop()

main(100, 0)


