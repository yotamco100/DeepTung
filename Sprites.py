import pygame
import math
GAME_BALL = "Extras\\Game_ball.png"
GRAVITY = 9.8
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLACK_PIXEL = (0, 0, 0, 1)
CYAN = (72, 118, 255)
LIGHT_GRAY = (192, 192, 192)


class Player1B(pygame.sprite.Sprite):
    def __init__(self, x, y, vel, angle, color, left, right, screen_relation=(1, 1)):
        super(self.__class__, self).__init__()
        self.color = color
        self.img = pygame.image.load(GAME_BALL).convert()
        screen_relation[0] *= self.img.get_width()
        screen_relation[1] *= self.img.get_height()
        self.img = pygame.transform.scale(self.img, screen_relation)
        array = pygame.PixelArray(self.img)
        array.replace(WHITE, self.color)
        self.img = array.make_surface()
        rot_image = pygame.transform.rotate(self.img, angle)
        self.rect = rot_image.get_rect()
        rot_rect = self.rect.copy()
        rot_rect.center = rot_image.get_rect().center
        self.image = rot_image.subsurface(rot_rect).copy()
        if self.color != BLACK_PIXEL:
            self.image.set_colorkey(BLACK_PIXEL)
        self.mask = pygame.mask.from_surface(self.image)
        self.rect = self.image.get_rect()
        self.rect.centerx = x
        self.rect.centery = y
        self.mask_outline = [(t[0] + self.rect.x, t[1] + self.rect.y)
                             for t in self.mask.outline()]
        self.x = float(self.rect.x)
        self.y = float(self.rect.y)
        self.starting_vel = vel
        self.angle = angle
        self.__vx = vel * math.cos(math.radians(self.angle))
        self.__vy = vel * math.sin(math.radians(self.angle))
        self.binds = (left, right)
        self.blank_line = False

    def update_image(self):
        x, y = self.rect.center
        self.image = pygame.transform.rotate(self.img, self.angle)
        if self.color != BLACK_PIXEL:
            self.image.set_colorkey(BLACK_PIXEL)
        self.mask = pygame.mask.from_surface(self.image)
        self.mask_outline = self.mask.outline()
        self.rect = self.image.get_rect()
        self.rect.center = (x, y)
        self.mask_outline = [(t[0] + self.rect.x, t[1] + self.rect.y)
                             for t in self.mask.outline()]

    def change_blank_line(self):
        self.blank_line = not self.blank_line

    def update_color(self, color):
        self.color = color
        self.update_image()

    def update_starting_vel(self, starting_vel):
        self.starting_vel = starting_vel

    def update_a(self, num):
        self.angle += num
        self.angle %= 360
        self.update_image()

    def update_v_a(self):
        self.__vx = self.starting_vel
        self.__vy = -self.starting_vel
        self.__vx *= math.cos(math.radians(self.angle))
        self.__vy *= math.sin(math.radians(self.angle))
        self.__vx /= 50.0
        self.__vy /= 50.0
        self.__vx = round(self.__vx, 4)
        self.__vy = round(self.__vy, 4)

    def update_loc(self):
        self.x += self.__vx
        self.rect.x = round(self.x, 4)
        self.y += self.__vy
        self.rect.y = round(self.y, 4)

    def get_pos(self):
        return self.rect.topleft

    def get_blank_line(self):
        return self.blank_line

    def get_mid(self):
        return self.rect.center

    def get_angle(self):
        return self.angle

    def get_v(self):
        return self.__vx, self.__vy

    def get_starting_v(self):
        return self.starting_vel

    def get_image(self):
        self.update_image()
        return self.image

    def get_color(self):
        return self.color

    def get_binds(self):
        return self.binds

    def get_rect(self):
        return self.rect

    def to_string(self):
        string = "pos: " + str(self.get_pos()) + "\n"
        string += "vel: " + str(self.get_v()) + "\n"
        string += "angle: " + str(self.get_angle()) + "\n"
        string += "Color: " + str(self.color)
        return string
