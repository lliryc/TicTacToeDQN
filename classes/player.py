import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

class Player:
    def __init__(self, cross, name):
        if cross:
            self.mark = 1
        else:
            self.mark = -1
        self.name = name
        random.seed(a=None, version=2)
    def turn(self, field, i, j):
        field.put(self.mark, i,j)
        return field.get_max_min(i, j)

class RandomPlayer(Player):
    def turn(self, field):
        i,j = field.get_free_cell()
        field.put(self.mark, i,j)
        return field.get_max_min(i,j)

class MinMaxPlayer(Player):
    def turn(self, field):
        cells = field.get_free_cells()

        for i,j in cells:
            if field.get_max_min(i,j) * self.mark == field.size - 1:
                field.put(self.mark, i, j)
                return field.get_max_min(i,j)
        for i,j in cells:
            if field.get_max_min(i,j) * self.mark == -(field.size - 1):
                field.put(self.mark, i, j)
                return field.get_max_min(i,j)

        center = (field.size // 2, field.size // 2)

        if center in cells:
            field.put(self.mark, center[0], center[1])
            return field.get_max_min(center[0], center[1])

        corners = [(i,j) for (i,j) in((0,0),(0,field.size -1), (field.size -1, 0), (field.size -1, field.size -1)) if (i,j) in cells]

        corners_evals = [field.get_max_min(i,j) for i,j in corners]

        if len (corners_evals) > 0:
            imax = np.argmax(corners_evals)
            i,j = corners[imax]
            field.put(self.mark, i, j)
            return field.get_max_min(i,j)

        cells_evals = [field.get_max_min(i,j) for i,j in cells]
        imax = np.argmax(cells_evals)
        i,j = cells[imax]
        field.put(self.mark, i, j)
        return field.get_max_min(i, j)

class DqnPlayer(Player):

    def __init__(self, cross, name, field):
        Player.__init__(self,cross, name)
        self.memory = deque(maxlen=100)
        self.gamma = 0.85
        self.epsilon = 0.5#1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.005
        self.tau = .125
        self.field = field

        self.model = self.create_model(field.size)
        self.target_model = self.create_model(field.size)

    def create_model(self, size):
        model = Sequential()

        model.add(Dense(24, input_dim=size*size, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(size*size))
        model.compile(loss="mean_squared_error",
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return self.untranslate(*self.field.get_free_cell())
        action = np.argmax(self.model.predict(state)[0])
        return action

    def translate(self, action):
        i = action // self.field.size
        j = action % self.field.size
        return i,j

    def untranslate(self, i,j):
        return i * self.field.size + j

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 18
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            try:
                state, action, reward, new_state, done = sample
                target = self.target_model.predict(state)
                if done:
                    target[0][action] = reward
                else:
                    Q_future = max(self.target_model.predict(new_state)[0])
                    target[0][action] = reward + Q_future * self.gamma
                self.model.fit(state, target, epochs=1, verbose=0)
            except Exception as e1:
                raise e1

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

    def reward(self, i, j):
        # if self.field.get_max_min(i, j) * self.mark == (self.field.size - 1):
        #     return 10 / self.field.turns
        # if self.field.get_max_min(i, j) * self.mark ==  - (self.field.size - 1):
        #     return 0.5
        #
        # center = (self.field.size // 2, self.field.size // 2)
        #
        # if (i, j) == center:
        #     return 0.5
        #
        #
        # corners = [(0,0),(0,self.field.size -1), (self.field.size -1, 0), (self.field.size -1, self.field.size -1)]
        #
        # if (i, j) in corners:
        #     return 0.3

        return 0.1




