# Author: Yotam Cohen
# An Achtung learning agent using Keras-TensorFlow tools.
from keras.models import Sequential
from keras.layers import Activation, Input, Dropout, MaxPooling2D, Dense, Flatten, Conv2D, Add, Concatenate, TimeDistributed, LeakyReLU, BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import Model
from keras.utils import plot_model
from keras.callbacks import CSVLogger
from keras.preprocessing.sequence import TimeseriesGenerator
from keras import losses
import numpy as np
from collections import deque
import random
#from PyGamePlayer.pygame_player import PyGamePlayer
from pygame.constants import K_a, K_d
#import cv2
#import Achtung2
#import keyboard_interaction
import keras.backend as K
#import cntk
import pygame
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time, sleep
import math
from threading import Thread
import sys

WINWIDTH = 1280  # width of the program's window, in pixels
WINHEIGHT = 720  # height in pixels
VK_KEY_A = 0x41
VK_KEY_D = 0x44
VK_LEFT = 0x42
VK_RIGHT = 0x56

np.set_printoptions(threshold=sys.maxsize, linewidth=np.nan)
BATCH_SIZE = 32
img_rows , img_cols = 1,48
img_channels = 48

class DQNAgent():
    # A Deep Q-Learning agent
    def __init__(self, state_size, hidden_sizes, action_size):
        # Agent constructor
        self.state_size = state_size  # input layer size
        self.action_size = action_size  # output layer size
        self.memory = deque(maxlen=50000)  # output history was 50k, changed to 5k after paper achtung.pdf adam lilja
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 5e-4
        self.momentum = 1
        self.model = self._build_model()
        self.epoch = 0
        with open('plot1.csv', 'w') as f:
            f.write('epoch,loss\n')
        with open('plot2.csv', 'w') as f:
            f.write('epoch,live_frames\n')

    def _build_model(self):
        # Neural Net for Deep Q-Learning model
        input1 = Input(shape=(img_cols,))
        model1 = Dense(img_cols, init='lecun_uniform')(input1)
        model1 = Activation('relu')(model1)
        """
        model1 = Dense(64)(model1)
        model1 = Activation('relu')(model1)
        model1 = Dense(32)(model1)
        model1 = Activation('relu')(model1)
        
        model1 = Dense(16)(model1)  # experimental
        model1 = Activation('relu')(model1)  # experimental
        """  # EXPERIMENTAL AF
        model1 = Dense(512, init='lecun_uniform')(model1)
        model1 = Activation('relu')(model1)
        model1 = Dropout(0.2)(model1)
        model1 = Dense(512, init='lecun_uniform')(model1)
        model1 = Activation('relu')(model1)
        model1 = Dropout(0.2)(model1)
        
        model1 = Dense(self.action_size, init='lecun_uniform')(model1)
        # EXPERIMENTAL AF
        outputs = Activation('linear')(model1)
        
        final = Model(inputs=input1, outputs=outputs)
        final.compile(loss=losses.mean_squared_error, optimizer=Adam())  # , decay=self.momentum))
        return final

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        print("act_values:", act_values)
        return np.argmax(act_values[0])

    def replay(self, batch_size, live_frames):
        
        print("replaying")
        minibatch = random.sample(self.memory, batch_size)
        f = open('plot1.csv', 'a')
        if self.epoch > 2:
            f2 = open('plot2.csv', 'a')
            f2.write(str(self.epoch) + ',' + str(live_frames) + '\n')
            f2.close()
        states = []
        targets = []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f)

        states = np.asarray(states).reshape(batch_size,img_cols)
        targets = np.asarray(targets).reshape(batch_size,3)
        history = self.model.train_on_batch(states, targets)
        print(history)
        f.write(str(self.epoch) + ',' + str(history) + '\n')
        self.epoch += 1
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        f.close()

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
   

class AchtungPlayer():
    def __init__(self, state_size, hidden_sizes, action_size, player):
        self.agent = DQNAgent(state_size+1, hidden_sizes, action_size)
        self._last_state = None
        self._last_action = 0
        self.state = []
        self.epochs = 0
        self.position = (0,0)
        self.angle = 0
        self.totalreward = 0
        self.player = player
        self.live_frames = 0

    # Step 1 (make an arm)
    def make_sonar_arm(self, x, y):
        spread = 10 # Default spread.
        distance = 1 # Gap before first sensor.
        arm_points = []
        # Make an arm. We build it flat because we’ll rotate it about 
        # the center later.
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))
        return arm_points
    # Steps 2 and 3 (rotate points, get readings, return distance):

    def get_arm_distance(self, arm, x, y, angle, offset, screen):
        show_sensors = False
        # Used to count the distance (iteration=distance).
        i = 0
        # Look at each point and see if we’ve hit something.
        for point in arm:
            i+=1
            # Move the point to the right spot.
            # We move one point at a time to save time if we
            # run into something. No sense rotating the whole arm.  
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], math.radians(90-angle+offset)
            )
            # Check if we’ve hit something. Return the current i
            # if we did.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= WINWIDTH-3 or rotated_p[1] >= WINHEIGHT-3:
                return i # Sensor is off the screen.
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i
            if show_sensors:
                pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (rotated_p), 2)
                pygame.display.update()
        # If we made it here, we didn't run into anything.
        return i
        # Step 4 is self-explanatory: just call these methods multiple times.

    def get_track_or_not(self, reading):
        if reading == pygame.color.THECOLORS['black']:
            return 0
        else:
            return 1

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_2 - y_1) * math.cos(radians) - \
            (x_2 - x_1) * math.sin(radians)
        new_x = x_change + x_1
        new_y = y_change + y_1
        return int(new_x), int(new_y)

    def get_sensor_readings(self, x, y, angle, screen_array):
        show_sensors = False
        # Set a default distance.
        distance = 5

        # Get the points, as if the angle is 0.
        # We use a list because it retains order.
        sens_points = []

        # Let's try making it a big grid.
        columns = 16
        rows = columns*2
        for j in (range(-columns,columns)):
            for i in range(rows):
                """
                if (i == 0 and (j == 0 or j == 1 or j == -1)) or \
                        (i == 1 and j == 0):
                    continue  # Skip the dots on top of the car.
                """
                sens_points.append((x+(distance*j), y+(i*distance)))

        # Now rotate those to make it in the front of the car.
        # And get the observations.
        sensor_obs = []
        for point in sens_points:
            # Get the point location.
            rotated_p = self.get_rotated_point(x, y, point[0], point[1], math.radians(90-angle))
            # Get the color there.
            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= WINWIDTH-125 or rotated_p[1] >= WINHEIGHT-3:
                sensor_obs.append(1)  # Sensor is off the screen.
            else:
                obs = screen_array.get_at(rotated_p)
                if obs == pygame.color.THECOLORS["black"]:
                    obs = 0
                else:
                    obs = 1
                sensor_obs.append(obs)
            # Now that we have the color, draw so we can see.
            if show_sensors:
                pygame.draw.circle(pygame.display.get_surface(), (255, 255, 255), (rotated_p), 2)
                pygame.display.update()
        return sensor_obs


    def get_keys_pressed(self, screen_array, reward, terminal):
        #print("acting")
        """
        screen_array = cv2.cvtColor(screen_array, cv2.COLOR_BGR2GRAY)
        # set the pixels to all be 0. or 1.
        _, screen_array = cv2.threshold(screen_array, 0, 1, cv2.THRESH_BINARY)
        """
        if not terminal:
            self.position = (self.player.x, self.player.y)
            self.angle = self.player.angle
        else:
            print("player dead, epoch:", self.epochs,"epsilon:", self.agent.epsilon)
        
        sonar_data = []
        for i in range(45):
            new_angle = math.radians(90-self.angle)
            new_position = [int(self.position[0]-2*5*math.sin(new_angle)), int(self.position[1]-2*5*math.cos(new_angle))]
            arm = self.make_sonar_arm(new_position[0], new_position[1])  # self.position[0], self.position[1])
            arm_distance = self.get_arm_distance(arm, new_position[0], new_position[1], self.angle-45, -6*i, screen_array)
            sonar_data.append(arm_distance)


        #self.state = self.get_sensor_readings(self.position[0], self.position[1], self.angle, screen_array)
        norm_data = [self.position[0], self.position[1], self.angle] + [(x)/39.0 for x in sonar_data]
        self.state = norm_data
        flip_state = 0
        #print("state:", self.state)
        """
        if self.epochs < 210:
            flip_state = random.randint(0,1)
        else:
            flip_state = 0
        
        if flip_state == 1:
            self.state = self.state[::-1]
        #print("flip?", flip_state)
        """
        if not terminal:
            self.live_frames += 1
            if len(self.state) == img_channels:
                # The entire segment above prepares the data for the network
                self.state = np.asarray([self.state], dtype=np.float32)  
                final_input = self.state
                if self._last_state is None:
                    self._last_state = final_input
                    if self._last_action != 2:  # if last action is not nothing, do the thing
                        if flip_state == 1:
                            self._reverse_angle_change_from_action(self._last_action)
                        else:
                            self._angle_change_from_action(self._last_action)
                    #else:
                        #print("no action")
                    self.state = []
                    
                    return self._key_presses_from_action(self._last_action)
                # Now it can't be the first frame, so we use it to train
                # print(sum(final_input[0]))
                #reward = np.sum(norm_data)
                #reward = np.median(norm_data)
                #print("reward:", reward)
                self.agent.remember(self._last_state, self._last_action, reward, final_input, terminal)
                
                action = self.agent.act(final_input)
                #print "{} action: ".format(self.key1), action, "epsilon:", self.agent.epsilon
                
                """ i don't know why this is here..? this does the last action every turn..?
                if action != 2:
                    self._angle_change_from_action(self._last_action)
                """

                self._last_action = action

                if action != 2:
                    if flip_state == 1:
                        self._reverse_angle_change_from_action(action)
                    else:
                        self._angle_change_from_action(action)
                #else:
                    #print("no action")

                self.state = []
                
                return "weird"
            # else
            
            return "weird but outside the loop"
        else:
            print("terminal")
            # NEW LINE, EXPERIMENTAL
            #print("reward:", reward)
            self.agent.remember(self._last_state, self._last_action, reward, self.state, terminal)
            self.epochs += 1
            self.totalreward = 0
            if len(self.state) == img_channels:
                del(self.state[0])
            if len(self.agent.memory) > BATCH_SIZE and self.epochs > 100:
                self.agent.replay(BATCH_SIZE, self.live_frames)
                if self.epochs % 100 == 0:
                    self.agent.save(".\\save\\player-checkpoint-{}.h5".format(self.epochs))
                """
                if self.epochs % 500 == 0:
                    self.agent.epsilon = 1.0
                """
                """
                if self.epochs % 100 == 0:
                    self.agent.learning_rate /= 2
                """
                self.live_frames = 0
            
            return self._key_presses_from_action(self._last_action)

    def get_feedback(self):
        if self.player.running == False:
            return (0, True) #If dead, punish
        return (1, False) #if nothing, give 0 point
        

    def _key_presses_from_action(self, action):
        if action == 0:
            return "left"
        if action == 1:
            return "right"
        return []
    
    def _key_presses_from_action3(self, action):
        if action == 0:
            return "left"  # left
        if action == 1:
            return "right"  # right
        return 7

    def _angle_change_from_action(self, action):
        if action == 0:
            #print("left")
            self.player.angle -= 10
            return
        if action == 1:
            #print("right")
            self.player.angle += 10

    def _reverse_angle_change_from_action(self, action):
        if action == 1:
            print("left")
            self.player.angle -= 10
            return
        if action == 0:
            print("right")
            self.player.angle += 10
            

def main():
    """
    # init env and the agent
    my_player = AchtungPlayer(1, [3], 3, VK_KEY_A, VK_KEY_D)
    plot_model(my_player.agent.model, to_file='model.png')
    #my_other_player = AchtungPlayer(1,[3], 3, VK_LEFT, VK_RIGHT)
    my_player.start()
    #my_player.agent.load(".\\save\\65-checkpoint-4700.h5")
    #my_other_player.start()
    Achtung2.main()
    """

if __name__ == "__main__":
    main()
