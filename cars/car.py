import pygame
import numpy as np


class Car:
    def __init__(
        self, x, y, sprite_path,
        sprite_size=(30, 60), angle=0, velocity=0, n_sensors=6
    ):
        self.x = x
        self.y = y
        self.velocity = velocity
        self.angle = angle
        self.angle = 0
        self.sprite_size = sprite_size
        self.n_sensors = n_sensors
        self.sprite = pygame.image.load(sprite_path)
        self.sprite = pygame.transform.scale(self.sprite, sprite_size)

        self.sensors_endpoints = [(x, y)] * n_sensors  # 5 car sensors
        self.sensors_readings = [0] * n_sensors

    def update_position(self, TRACK_MAP):
        car_position = (int(self.x), int(self.y))
        if TRACK_MAP[car_position[1], car_position[0]]:
            return

        alpha = (self.angle % 360) * np.pi / 180
        angle_v = 180 / (self.n_sensors - 1) * np.pi / 180

        for i in range(self.n_sensors):
            vx = np.cos(angle_v * i - alpha)
            vy = np.sin(angle_v * i - alpha)

            tmp_x, tmp_y = self.x, self.y

            while not TRACK_MAP[int(tmp_y + vy), int(tmp_x + vx)]:
                tmp_x += vx
                tmp_y += vy

            self.sensors_endpoints[i] = (tmp_x, tmp_y)
            self.sensors_readings[i] = np.sqrt(
                (self.x - tmp_x) ** 2 + (self.y - tmp_y) ** 2
            )

        self.x += np.sin(alpha) * self.velocity
        self.y += np.cos(alpha) * self.velocity

    def draw(self, win):
        car_position = (
            int(self.x),
            int(self.y)
        )

        oldRect = self.sprite.get_rect(center=car_position)
        rot_image = pygame.transform.rotate(self.sprite, self.angle)
        rot_rect = rot_image.get_rect(center=oldRect.center)

        win.blit(
            rot_image,
            dest=rot_rect
        )

        pygame.draw.circle(win, (0,   255, 0), car_position, 4)
        for x, y in self.sensors_endpoints:
            pygame.draw.line(win, (0,   0, 255), car_position, (x, y), 1)
