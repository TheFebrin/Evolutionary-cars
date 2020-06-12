import pygame
import numpy as np
import sys
from neural_network import nn
from functools import reduce
import torch
import time

sys.path.insert(0, '..')
CHECKPOINT_AWARD = 100000


class Car:
    def __init__(
        self, id, x, y, sprite_path,
        sprite_size=(30, 60), angle=0, velocity=0, n_sensors=6
    ):
        self.id = id
        self.x = x
        self.y = y
        self.velocity = velocity
        self.angle = angle
        self.sprite_size = sprite_size
        self.n_sensors = n_sensors
        self.sprite_size = sprite_size
        self.sprite = pygame.image.load(sprite_path)
        self.sprite = pygame.transform.scale(self.sprite, sprite_size)
        self.dead = True
        self.next_checkpoint = 0

        self.sensors_endpoints = [(x, y)] * n_sensors  # 5 car sensors
        self.sensors_readings = [0] * n_sensors

        self.network = nn.Network(
            in_dim=n_sensors + 2,  # n_sensors + act_velocity + act angle
            h1=4,
            h2=3,
            out_dim=2
        )
        self.n_parameters = sum(reduce(lambda a, b: a * b, x.size())
                                for x in self.network.parameters())

        self.genotype = np.random.normal(0, 0.1, self.n_parameters)
        self.objective_value = 1
        self.time_start = time.time()

    def predict_move(self):
        if self.dead:
            return

        assert(self.n_parameters == len(self.genotype))
        idx = 0
        with torch.no_grad():
            for name, p in self.network.named_parameters():
                if 'weight' in name:
                    w_size = p.shape[0] * p.shape[1]
                    w = self.genotype[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w).view(p.shape))
                elif 'bias' in name:
                    w_size = p.shape[0]
                    w = self.genotype[idx: idx + w_size]
                    p.copy_(torch.from_numpy(w))
                else:
                    raise ValueError('Unknown parameter name "%s"' % name)
                idx += w_size

        X = torch.Tensor(self.sensors_readings + [self.velocity, self.angle])
        preds = self.network(X.float()).detach().numpy()
        self.velocity += preds[0]
        self.angle += preds[1]

    def update_position(self, TRACK_MAP, CHECKPOINTS_MAPS, checkpoints):
        if self.dead:
            return

        car_position = (int(self.x), int(self.y))
        if TRACK_MAP[car_position[1], car_position[0]]:  # EVALUATE CAR
            self.dead = True
            nc = checkpoints[self.next_checkpoint]
            d = ((self.x - nc[0]) ** 2 + (self.y - nc[1]) ** 2)  # no sqrt
            self.objective_value -= d
            return

        if not CHECKPOINTS_MAPS[self.next_checkpoint][car_position[1], car_position[0]]:
            # print(f'CAR: {self.id} REACHED {self.next_checkpoint}')
            self.objective_value += CHECKPOINT_AWARD
            self.next_checkpoint += 1
            self.next_checkpoint %= 4

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
        car_position = (int(self.x), int(self.y))

        oldRect = self.sprite.get_rect(center=car_position)
        rot_image = pygame.transform.rotate(self.sprite, self.angle)
        rot_rect = rot_image.get_rect(center=oldRect.center)

        win.blit(rot_image, dest=rot_rect)

        font = pygame.font.Font(pygame.font.get_default_font(), 25)
        id = font.render(f'{self.id}', True, (0, 255, 0))
        goal = font.render(f'-{self.next_checkpoint}', True, (255, 255, 0))
        win.blit(id, dest=(car_position[0] - 12, car_position[1] - 12))
        win.blit(goal, dest=(car_position[0] + 12, car_position[1] - 12))

        if self.dead:
            return

        # pygame.draw.circle(win, (0,   255, 0), car_position, 4)
        for x, y in self.sensors_endpoints:
            pygame.draw.line(win, (0,   0, 255), car_position, (x, y), 1)
