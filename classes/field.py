import numpy as np
import random

class Field:
    def __init__(self, size: int):
        self.size = size
        self.turns = 0
        self.space = np.zeros((size, size), dtype=np.int)
        self.sys_random = random.SystemRandom()

    def reset(self):
        self.turns = 0
        self.space = np.zeros((self.size, self.size), dtype=np.int)

    def get(self, i: int, j: int)->int:
        if (i < 0 or i >= self.size) or (j < 0 or j > self.size):
            raise Exception("out of range")
        return self.space[i][j]

    def put(self, sign: int, i: int, j: int):
        if sign == 0:
            raise Exception("sign must be positive or negative")
        if self.space[i][j] != 0:
            raise Exception("cell already was filled")
        self.turns = self.turns + 1
        self.space[i][j] = np.sign(sign)

    def full(self):
        return self.turns == self.size * self.size

    def get_max_min(self, i: int, j: int):
        vsum = sum([self.space[i][k] for k in range(0, self.size)])
        hsum = sum([self.space[k][j] for k in range(0, self.size)])

        dmajorsum = 0
        if i == j:
            dmajorsum = sum([self.space[i1][i1] for i1 in range(0, self.size)])

        dminorsum = 0
        if (i + j) == (self.size - 1):
            dminorsum = sum([self.space[i1][self.size - i1 - 1] for i1 in range(0, self.size)])

        mx = np.amax([vsum, hsum, dminorsum, dmajorsum])
        mn = np.amin([vsum, hsum, dminorsum, dmajorsum])
        return mn if np.abs(mn) > np.abs(mx) else mx

    def get_max_min_rows(self):
        return [sum([self.space[i][k] for k in range(0, self.size)]) for i in range(0, self.size)]

    def get_max_min_cols(self):
        return [sum([self.space[k][j] for k in range(0, self.size)]) for j in range(0, self.size)]

    def get_max_min_diags(self):
        dminorsum = sum([self.space[i1][self.size - j1 - 1] for i1, j1 in zip(range(0, self.size), range(0, self.size))])
        dmajorsum = sum([self.space[i1][j1] for i1, j1 in zip(range(0, self.size), range(0, self.size))])
        return [dmajorsum, dminorsum]

    def get_free_cells(self):
        return [(i,j) for i in range(0, self.size) for j in range(0, self.size) if self.space[i][j] == 0]

    def get_free_row_cells(self, i):
        return [(i,j) for j in range(0, self.size) if self.space[i][j] == 0]

    def get_free_col_cells(self, j):
        return [(i, j) for i in range(0, self.size) if self.space[i][j] == 0]

    def get_free_major_diag_cells(self):
        pass

    def get_free_minor_diag_cells(self):
        pass

    def get_free_cell(self):
        free_cells = self.get_free_cells()
        res = self.sys_random.choice(free_cells)
        return res

    def print_all(self):
        for i in range(0, self.size):
            for j in range(0, self.size):
                print("%3s" % ('o' if self.space[i][j] == -1 else ('x' if self.space[i][j] == 1 else '-')), sep='', end='')
            print()

    def get_state(self):
        return np.copy(self.space)

    def set_state(self, state):
        self.space = state



