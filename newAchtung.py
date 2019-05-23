import random
import sys
import time
import pygame
import pygame.gfxdraw
import trial2
import os
from math import *
from pygame.locals import *
"""
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = "dummy"
"""
SPEED = 100000000       # frames per second setting
WINWIDTH = 1280  # width of the program's window, in pixels
WINHEIGHT = 720  # height in pixels
RADIUS = 5       # radius of the circles
PLAYERS = 2      # number of players

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
FUCHSIA = (255, 0, 255)
ORANGE = (255, 140, 0)
YELLOW = (255, 255, 0)

P1COLOUR = RED
P2COLOUR = GREEN
P3COLOUR = BLUE
P4COLOUR = FUCHSIA
P5COLOUR = ORANGE
P6COLOUR = YELLOW


def main():
    # main loop
    global FPS_CLOCK, SCREEN, DISPLAYSURF, MY_FONT
    pygame.init()
    FPS_CLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((WINWIDTH, WINHEIGHT))
    DISPLAYSURF = pygame.Surface(SCREEN.get_size())
    pygame.display.set_caption('DeepTung')
    # pygame.mixer.music.load('test.mp3')
    # pygame.mixer.music.play(-1, 0.0)
    MY_FONT = pygame.font.SysFont('bauhaus93', 37)
    while True:
        rungame()
        gameover()


class Player(object):
    # Class which can be used to generate random position and angle, to compute movement values and to draw player
    def __init__(self):
        self.running = True
        self.colour = None
        self.score = 0

    def gen(self):
        # generates random position and direction
        self.x = random.randrange(50, WINWIDTH - 165)
        self.y = random.randrange(50, WINHEIGHT - 50)
        self.angle = random.randrange(0, 360)

    def move(self):
        # computes current movement
        self.x += int(RADIUS * 2 * cos(radians(self.angle)))
        self.y += int(RADIUS * 2 * sin(radians(self.angle)))

    def draw(self):
        # drawing players
        pygame.gfxdraw.aacircle(DISPLAYSURF, self.x, self.y, RADIUS, self.colour)
        pygame.gfxdraw.filled_circle(DISPLAYSURF, self.x, self.y, RADIUS, self.colour)


def rungame():
    global WINNER
    DISPLAYSURF.fill(BLACK)
    #pygame.draw.aaline(DISPLAYSURF, WHITE, (WINWIDTH-100, 0), (WINWIDTH-100, WINHEIGHT))
    WINNER = []
    first = True
    run = True
    players_running = PLAYERS
    if PLAYERS == 3:
        max_score = 10
    else:
        max_score = 5

    # generating players
    player1 = Player()
    player2 = Player()
    player3 = Player()
    player4 = Player()
    player5 = Player()
    player6 = Player()
    player_t = [player1, player2, player3, player4, player5, player6]
    for i in range(PLAYERS):
        player_t[i].gen()

    bots = [trial2.AchtungPlayer(1, [3], 3, player1), trial2.AchtungPlayer(1, [3], 3, player2)]  # , trial2.AchtungPlayer(1, [3], 3, player3), trial2.AchtungPlayer(1, [3], 3, player4), trial2.AchtungPlayer(1, [3], 3, player5), trial2.AchtungPlayer(1, [3], 3, player6)]
    bots[1].agent = bots[0].agent
    #bots[2].agent = bots[0].agent
    #bots[3].agent = bots[0].agent
    #bots[4].agent = bots[0].agent
    #bots[5].agent = bots[0].agent
    #bots[1].agent.learning_rate = 0.01
    """
    bots[0].agent.load('../PROJECT DEEPTUNG SUCCESS ATTEMPT MAYBE/player-checkpoint-400.h5')
    bots[0].epochs = 235
    bots[0].agent.epsilon = 0.5
    bots[0].agent.epoch = 135
    """
    player_epochs=[False, False, False, False, False, False]
    while run:
        # initializing players colours
        player1.colour = P1COLOUR
        player2.colour = P2COLOUR
        player3.colour = P3COLOUR
        player4.colour = P4COLOUR
        player5.colour = P5COLOUR
        player6.colour = P6COLOUR

        # generating random holes
        hole = random.randrange(1, 20)
        if hole == 3:
            player1.move()
            player1.colour = BLACK
        elif hole == 5:
            player2.move()
            player2.colour = BLACK
        """
        elif hole == 7:
            player3.move()
            player3.colour = BLACK
        elif hole == 9:
            player4.move()
            player4.colour = BLACK
        elif hole == 11:
            player5.move()
            player5.colour = BLACK
        elif hole == 13:
            player6.move()
            player6.colour = BLACK
        """
        for i in range(PLAYERS):  # loop for checking positions, drawing, moving and scoring for all players
            if player_t[i].running:
                if player_t[i].angle < 0:
                    player_t[i].angle += 360
                elif player_t[i].angle >= 360:
                    player_t[i].angle -= 360

                # checking if someone fails
                if (player_t[i].x > WINWIDTH-3 or player_t[i].x < 3 or
                            player_t[i].y > WINHEIGHT-3 or player_t[i].y < 3 or
                            DISPLAYSURF.get_at((player_t[i].x, player_t[i].y)) != BLACK):
                    print("player hit the wall:", (player_t[i].x > WINWIDTH-3 or player_t[i].x < 3 or player_t[i].y > WINHEIGHT-3 or player_t[i].y < 3))
                    player_t[i].running = False
                    players_running -= 1

                player_t[i].draw()
                player_t[i].move()

        for event in pygame.event.get():
            if event.type == QUIT:
                shutdown()
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    shutdown()

        # steering
        # keys = pygame.key.get_pressed()
        for i in range(len(bots)):
            if not player_epochs[i]:
                reward, terminal = bots[i].get_feedback()
                bots[i].get_keys_pressed(pygame.display.get_surface(), reward, terminal)
                if terminal:
                    player_epochs[i] = True
        # bot decision making
        """
        if keys[pygame.K_LEFT]:
            player1.angle -= 10
        if keys[pygame.K_RIGHT]:
            player1.angle += 10
        """

        # drawing scores
        #scoring(player1.score, P1COLOUR)

        # drawing all on the screen
        SCREEN.blit(DISPLAYSURF, (0, 0))
        pygame.display.update()

        # checking if someone reach max score and win
        if players_running == 0:
            #pygame.time.wait(1000)
            DISPLAYSURF.fill(BLACK)
            #pygame.draw.aaline(DISPLAYSURF, WHITE, (WINWIDTH-110, 0), (WINWIDTH-110, WINHEIGHT))
            first = True
            players_running = PLAYERS
            for i in range(PLAYERS):
                player_t[i].gen()
                player_t[i].running = True
            player_epochs=[False, False, False, False, False, False]
            continue

        if first:  # if the game starts, wait some time
            #pygame.time.wait(1500)
            first = False

        FPS_CLOCK.tick(SPEED)


def scoring(play1score, colour1):
    # drawing scores
    colour0 = WHITE
    score_msg = MY_FONT.render("Score:", 1, colour0, BLACK)
    score1_msg = MY_FONT.render("P1: " + str(play1score), 1, colour1, BLACK)
    DISPLAYSURF.blit(score_msg, (WINWIDTH - 110, WINHEIGHT/10))
    DISPLAYSURF.blit(score1_msg, (WINWIDTH - 108, WINHEIGHT/10 + 40))


def gameover():
    # drawing winner/s and waiting for key press
    if len(WINNER) == 1:
        end_msg = "Player %d wins! Press button to go to main menu." % WINNER[0]
    elif len(WINNER) == 2:
        end_msg = "Players %d and %d ties! Press button to go to main menu." % (WINNER[0], WINNER[1])
    end_msg_render = MY_FONT.render(end_msg, 1, WHITE, BLACK)
    SCREEN.blit(end_msg_render, ((WINWIDTH - MY_FONT.size(end_msg)[0]) / 2, WINHEIGHT/5))
    pygame.display.update()
    end = True
    while end:
        for event in pygame.event.get():
            if event.type == QUIT:
                shutdown()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    shutdown()
                else:
                    end = False
        FPS_CLOCK.tick(SPEED)

def shutdown():
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()