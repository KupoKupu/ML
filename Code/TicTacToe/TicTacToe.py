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
p_train = False
v_train = False

size = 6
win_size = 4

b = [[0] * size for i in range(size)]

states = torch.zeros(size, size).to(device)
# state1 = torch.ones_like(states).to(device)
# state2 = torch.ones_like(states).to(device) * 2

Traj1 = []
Traj2 = []

class action(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(1, 64, 4)
        self.norm1 = nn.LayerNorm([64, size - 3, size - 3])
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.norm2 = nn.LayerNorm([256, size - 5, size - 5])
        # self.conv3 = nn.Conv2d(128, 256, 4)
        # self.norm3 = nn.LayerNorm([256, 1, 1])
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, size ** 2)
        self.Softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.norm1(self.relu(self.conv1(x.unsqueeze(0))))
        x = self.norm2(self.relu(self.conv2(x))).squeeze()
        # x = self.norm3(self.relu(self.conv3(x))).squeeze()
        x = self.norm3(self.relu(self.fc1(x)))
        x = self.norm4(self.relu(self.fc2(x)))
        x = self.Softmax(self.fc3(x))
        return x

class V_net(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.conv1 = nn.Conv2d(1, 64, 4)
        self.norm1 = nn.LayerNorm([64, size - 3, size - 3])
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.norm2 = nn.LayerNorm([256, size - 5, size - 5])
        # self.conv3 = nn.Conv2d(128, 256, 4)
        # self.norm3 = nn.LayerNorm([256, 1, 1])
        self.norm3 = nn.LayerNorm(256)
        self.norm4 = nn.LayerNorm(256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.norm1(self.relu(self.conv1(x.unsqueeze(0))))
        x = self.norm2(self.relu(self.conv2(x))).squeeze()
        # x = self.norm3(self.relu(self.conv3(x))).squeeze()
        x = self.norm3(self.relu(self.fc1(x)))
        x = self.norm4(self.relu(self.fc2(x)))
        x = self.tanh(self.fc3(x))
        return x

Kupo = action(size).to(device)
Bot = action(size).to(device)
V = V_net(size).to(device)

optimizer_A2 = torch.optim.Adam(Kupo.parameters(), lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_V = torch.optim.Adam(V.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_A1 = torch.optim.Adam(Bot.parameters(), lr=0.0001, weight_decay=0.01, betas=(0.9, 0.999))
optimizer_MCTS = torch.optim.Adam(Kupo.parameters(), lr=0.001, weight_decay=0.01, betas=(0.9, 0.999))

def p1_move(r, c):
    global states
    global b

    b[r][c].configure(text="X")
    states[r][c] = 1


def bot_move():
    global states
    global Traj1

    legal_mask = (states == 0).view(-1).to(device)

    assert not torch.isnan(Bot(states)).any()

    idx = Categorical(Bot(states) * legal_mask).sample()
    m, n = idx // size, idx % size
    states[m][n] = 1
    current_state = states.clone().to(device)
    current_traj = [current_state, idx]
    Traj1.append(current_traj)



def p2_move(mcts:bool):
    global states
    global b
    global Traj2

    if mcts:
        mct = MCTS(states, 0.5, 20)
        idx = mct.output()
    else:
        legal_mask = (states == 0).view(-1).to(device)
        assert not torch.isnan(Kupo(states)).any()
        idx = Categorical(Kupo(states) * legal_mask).sample()

    m, n = idx // size, idx % size
    if not p_train and not v_train:
        b[m][n].configure(text="O")
    states[m][n] = 2
    current_state = states.clone().to(device)
    current_traj = [current_state, idx]
    Traj2.append(current_traj)

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
    global Traj1
    global Traj2
    global p_train
    global v_train
    global states

    if not p_train and not v_train:
        if ending == 3:
            messagebox.showinfo("tie", "Tie")
        else:
            messagebox.showinfo("Winner", f"Player {ending}" + " Won")
    elif p_train and not v_train:
        Traj1.append([torch.ones_like(states).to(device) * ending, 0])
        action_update(1, Bot, optimizer_A1)
        Traj2.append([torch.ones_like(states).to(device) * ending, 0])
        action_update(2, Kupo, optimizer_A2)
        if ending == 1:
            print("Bot won!")
        elif ending == 2:
            print("Kupo won!")
        elif ending == 3:
            print('Tie!')
    elif v_train and not p_train:
        Traj2.append([torch.ones_like(states).to(device) * ending, 0])
        value_update()



def Reward(s, a, player):
    global Traj1
    global Traj2

    if player == 1:
        traj = Traj1
    else:
        traj = Traj2

    R = 0
    # m, n = a // size, a % size
    # s_a = s.clone()
    # s_a[m][n] = player
    # p_counts = [check_consecutive(s_a, i, player) for i in range(2, win_size)]
    # # R += 16 * p_counts[2] + 8 * (p_counts[1] - 2 * p_counts[2]) + 4 * (p_counts[0] - 2 * p_counts[1] - 3 * p_counts[2])
    # R += 16 * p_counts[1] + 8 * (p_counts[0] - 2 * p_counts[1])
    #
    # o_counts = [check_consecutive(s_a, i, 3 - player) for i in range(2, win_size)]
    # # R -= 16 * o_counts[2] + 8 * (o_counts[1] - 2 * o_counts[2]) + 4 * (o_counts[0] - 2 * o_counts[1] - 3 * o_counts[2])
    # R += 16 * o_counts[1] + 8 * (o_counts[0] - 2 * o_counts[1])
    #
    # if check_consecutive(s_a, win_size, player) > 0:
    #     R += 100
    if torch.equal(traj[-1][0], torch.ones_like(states).to(device) * (3 - player)):
        R -= 1
    if torch.equal(traj[-1][0], torch.ones_like(states).to(device) * player):
        R += 1
    return R

def action_update(player, A, optim):
    global Traj1
    global Traj2

    if player == 1:
        traj = Traj1
    else:
        traj = Traj2

    optim.zero_grad()
    l = 0
    for node in traj:
        l -= torch.log(A(node[0])[node[1]]) * Reward(node[0], node[1], player)
    print(player, 'l:', l)
    l.backward()
    optim.step()

def value_update():
    global Traj2

    u = 0
    if torch.equal(Traj2[-1][0], torch.ones_like(states).to(device)):
        u = -1
    elif torch.equal(Traj2[-1][0], torch.ones_like(states).to(device) * 2):
        u = 1

    optimizer_V.zero_grad()
    l = 0
    for node in Traj2:
        l += (V(node[0]) - u) ** 2 / 2
    print('l:', l)
    l.backward()
    optimizer_V.step()


def load():
    file_path = Path('64checkpoint_A1.pth')
    if file_path.is_file():
        cp = torch.load('64checkpoint_A1.pth')
        optimizer_A1.load_state_dict(cp['64optimizer_state_dict_A1'])
        Bot.load_state_dict(cp['64model_state_dict_A1'])

    file_path = Path('64checkpoint_A2.pth')
    if file_path.is_file():
        cp = torch.load('64checkpoint_A2.pth')
        optimizer_A2.load_state_dict(cp['64optimizer_state_dict_A2'])
        Kupo.load_state_dict(cp['64model_state_dict_A2'])

    file_path = Path('64checkpoint_V.pth')
    if file_path.is_file():
        cp = torch.load('64checkpoint_V.pth')
        V.load_state_dict(cp['64model_state_dict_V'])
        optimizer_V.load_state_dict(cp['64optimizer_state_dict_V'])

    file_path = Path('64checkpoint_MCTS.pth')
    if file_path.is_file():
        cp = torch.load('64checkpoint_MCTS.pth')
        Kupo.load_state_dict(cp['64model_state_dict_MCTS'])
        optimizer_MCTS.load_state_dict(cp['64optimizer_state_dict_MCTS'])


def save():
    cp = {
        '64model_state_dict_A1': Bot.state_dict(),
        '64optimizer_state_dict_A1': optimizer_A1.state_dict(),
    }
    torch.save(cp, '64checkpoint_A1.pth')

    cp = {
        '64model_state_dict_A2': Kupo.state_dict(),
        '64optimizer_state_dict_A2': optimizer_A2.state_dict(),
    }
    torch.save(cp, '64checkpoint_A2.pth')

    cp = {
        '64model_state_dict_V': V.state_dict(),
        '64optimizer_state_dict_V': optimizer_V.state_dict(),
    }
    torch.save(cp, '64checkpoint_V.pth')

    cp = {
        '64model_state_dict_MCTS': Kupo.state_dict(),
        '64optimizer_state_dict_MCTS': optimizer_MCTS.state_dict(),
    }
    torch.save(cp, '64checkpoint_MCTS.pth')


def train_loop(rounds, p_or_v:str):
    global stop_game
    global Traj1
    global Traj2
    global states
    global p_train
    global v_train

    if p_or_v == 'p':
        p_train = True
        v_train = False
        Kupo.train()
        Bot.train()

    if p_or_v == 'v':
        p_train = False
        v_train = True
        V.train()

    bot_win = 0
    kupo_win = 0

    load()

    for i in range(rounds):
        print('Round ' + str(i + 1))
        stop_game = False
        states = torch.zeros(size, size).to(device)
        Traj1 = []
        Traj2 = []

        p2_move(mcts=False)

        while not stop_game:
            check_tie()
            if stop_game:
                update(3)
                break

            bot_move()
            check_if_win(1)
            if stop_game:
                update(1)
                bot_win += 1
            else:
                check_tie()
                if stop_game:
                    update(3)
                    break
                p2_move(mcts=False)
                check_if_win(2)
                if stop_game:
                    update(2)
                    kupo_win += 1

    save()

    print('bot won', bot_win, 'out of', rounds, 'rounds' )
    print('kupo won', kupo_win, 'out of', rounds, 'rounds')

def train_mcts(rounds):
    global stop_game
    global Traj1
    global Traj2
    global states
    global p_train

    p_train = True


    bot_win = 0
    kupo_win = 0

    load()
    Kupo.train()

    for i in range(rounds):
        print('Round ' + str(i + 1))
        stop_game = False
        states = torch.zeros(size, size).to(device)
        Traj1 = []
        Traj2 = []

        optimizer_MCTS.zero_grad()
        legal_mask = (states == 0).view(-1).to(device)
        prob = Kupo(states) * legal_mask
        prob_a = prob
        mcts = MCTS(states, 0.5, 20)
        prob_mcts = mcts.output_n() / torch.max(mcts.output_n())
        l = nn.CrossEntropyLoss()(prob_mcts, prob_a)
        l.backward()
        print("l:", l.item())
        optimizer_MCTS.step()

        p2_move(mcts=False)


        while not stop_game:
            check_tie()
            if stop_game:
                break

            bot_move()
            check_if_win(1)
            if stop_game:
                bot_win += 1
            else:
                check_tie()
                if stop_game:
                    break

                optimizer_MCTS.zero_grad()
                legal_mask = (states == 0).view(-1).to(device)
                prob = Kupo(states) * legal_mask
                prob_a = prob
                mcts = MCTS(states, 0.5, 20)
                prob_mcts = mcts.output_n() / torch.max(mcts.output_n())
                l = nn.CrossEntropyLoss()(prob_mcts, prob_a)
                l.backward()
                print("l:", l.item())
                optimizer_MCTS.step()

                p2_move(mcts=False)
                check_if_win(2)
                if stop_game:
                    kupo_win += 1

    save()

    print('bot won', bot_win, 'out of', rounds, 'rounds' )
    print('kupo won', kupo_win, 'out of', rounds, 'rounds')

def move(r, c):
    global Traj2

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
            p2_move(mcts=True)
            check_if_win(2)
            if stop_game:
                update(2)
            else:
                check_tie()
                if stop_game:
                    update(3)

class MCTS:
    def __init__(self, s, eta, n):
        self.states = s.clone()
        self.Q = torch.zeros(size ** 2).to(device)
        self.N = torch.zeros(size ** 2).to(device)
        self.eta = eta
        self.stop = False
        self.times = n

    def check_if_win(self, s, player):
        if check_consecutive(s, win_size, player) > 0:
            self.stop = True

    def check_tie(self, s):
        if all([all(row) for row in s]):
            self.stop = True

    def fast_rollout(self):
        s = self.states.clone()
        score = (self.Q + self.eta * Kupo(s)) / (1 + self.N)
        legal_mask = (s != 0).view(-1).to(device)
        a = torch.argmax(torch.Tensor.masked_fill(score, legal_mask, -torch.inf))
        self.N[a] += 1
        m, n = a // size, a % size
        s[m][n] = 2
        self.check_if_win(s, player=2)
        if self.stop:
            self.Q[a] += (V(s).item() + 1) / 2
        self.check_tie(s)
        if self.stop:
            self.Q[a] += V(s).item() / 2
        if not self.stop:
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(Bot(s) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = 1
            value = V(s).item()
            self.check_if_win(s, player=1)
            if self.stop:
                self.Q[a] += (value - 1) / 2
            self.check_tie(s)
            if self.stop:
                self.Q[a] += value / 2
        while not self.stop:
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(Kupo(s) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = 2
            self.check_if_win(s, player=2)
            if self.stop:
                self.Q[a] += (value + 1) / 2
                break
            self.check_tie(s)
            if self.stop:
                self.Q[a] += value / 2
                break
            legal_mask = (s == 0).view(-1).to(device)
            idx = Categorical(Bot(s) * legal_mask).sample()
            m, n = idx // size, idx % size
            s[m][n] = 1
            self.check_if_win(s, player=1)
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
        out = torch.argmax(self.N)
        return out

    def output_n(self):
        for i in range(self.times):
            self.fast_rollout()
        return self.N


root = Tk()
root.title("Kupo\'s TicTacToe")

def new_game():
    global train
    train = False
    Kupo.train()

    for i in range(size):
        for j in range(size):
            b[i][j] = Button(
                height=24 // size, width=48 // size + 1,
                font=("Helvetica", "20"),
                command=lambda r=i, c=j: move(r, c)
                )
            b[i][j].grid(row=i, column=j)

    p2_move(mcts=True)

def main(rounds, tr):
    if tr == 1:
        train_loop(rounds, 'p')
    elif tr == 2:
        train_loop(rounds, 'v')
    elif tr == 3:
        train_mcts(rounds)
    else:
        load()
        new_game()
        save()
        root.mainloop()

def loop(r1, r2, r3):
    main(r1, 1)
    main(r2, 2)
    # main(r3, 3)

loop(5000, 5000, 50)

# main(10, 0)


