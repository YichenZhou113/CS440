import utils
import numpy as np
import random
import math

class Agent:

    def __init__(self, actions, two_sided = False):
        self.two_sided = two_sided
        self._actions = actions
        self._train = True
        self._x_bins = utils.X_BINS
        self._y_bins = utils.Y_BINS
        self._v_x = utils.V_X
        self._v_y = utils.V_Y
        self._paddle_locations = utils.PADDLE_LOCATIONS
        self._num_actions = utils.NUM_ACTIONS
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.SApair = utils.create_q_table()
        self.prev_state = [6,6,1,1,6]
        self.prev_action = 0
        self.prev_bounces = 0
        self.dis_fac = 0.8
        self.lrate = 6
        self.epsilon = 0.6

    def act(self, state, bounces, done, won):

        x_pos = state[0]
        y_pos = state[1]
        unit_pos = 1/12
        x_bin = math.floor(x_pos/unit_pos)
        y_bin = math.floor(y_pos/unit_pos)

        if x_pos >= 1:
            x_bin = 11
        if y_pos >= 1:
            y_bin = 11

        if state[2]<0:
            x_velo = 0
        else:
            x_velo = 1

        if state[3] > 0.015:
            y_velo = 2
        elif state[3] < -0.015:
            y_velo = 0
        else:
            y_velo = 1

        if state[4] >= 0.8:
            d_paddle = 11
        else:
            d_paddle = math.floor(12 * state[4] / 0.8)

        if self.two_sided == False:
            if done:
                self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 1] = -1
                return 0
            elif bounces > self.prev_bounces:
                cur_reward = 1
            else:
                cur_reward = 0


        if self.two_sided == True:
            if won:
                self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 1] = 30
                return 0
            elif done:
                self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 1] = -1
                return 0
            elif bounces > self.prev_bounces:
                cur_reward = 1
            else:
                cur_reward = 0

        #print(x_bin, y_bin, x_velo, y_velo, d_paddle, done, bounces)

        if self.prev_action == -1:
            action_index = 0
        elif self.prev_action == 0:
            action_index = 1
        else:
            action_index = 2

        #print(x_pos, y_pos, x_bin,y_bin, d_paddle, done)
        Q0 = self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 0]
        Q1 = self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 1]
        Q2 = self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 2]
        max_Q = max(self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 0], self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 1], self.Q[x_bin, y_bin, x_velo, y_velo, d_paddle, 2])
        #print('QQQQQQ',Q0,Q1,Q2)

        if self._train == False:
            if(Q2 > Q1 and Q2 > Q0):
                return 1
            elif(Q2 > Q0 and Q1 > Q2):
                return 0
            else:
                return -1

        if (random.random()>self.epsilon):
            if (Q2 > Q1 and Q2 > Q0):
                next_action = 1
            elif (Q1 >= Q0 and Q1 >= Q2):
                next_action = 0
            else:
                next_action = -1
        else:
            next_action = random.choice([-1,0,1])

        #print(self.prev_state[4])
        self.Q[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], self.prev_state[4], action_index] += \
             self.lrate / (self.lrate + self.SApair[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], self.prev_state[4], action_index]) \
              * (cur_reward - self.Q[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], self.prev_state[4], action_index] \
              + self.dis_fac * max_Q)

        #print(self.Q[self.prev_state[0], self.prev_state[1], self.prev_state[2], self.prev_state[3], self.prev_state[4], action_index])

        self.prev_state = [x_bin, y_bin, x_velo, y_velo, d_paddle]
        self.prev_action = next_action
        self.prev_bounces = bounces
        self.SApair[x_bin, y_bin, x_velo, y_velo, d_paddle, next_action] += 1
        return next_action

    def train(self):
        self._train = True

    def eval(self):
        self._train = False

    def save_model(self,model_path):
        # At the end of training save the trained model
        utils.save(model_path,self.Q)

    def load_model(self,model_path):
        # Load the trained model for evaluation
        self.Q = utils.load(model_path)
