from PIL import Image
import pygame
import numpy as np
import time

from cars import car


def load_racing_tracks(k):
    track1 = pygame.image.load('images/track1.jpg')
    track1BW = Image.open('images/track1BW.jpg').convert('1')
    track2 = pygame.image.load('images/track2.jpg')
    track2BW = Image.open('images/track2BW.jpg').convert('1')
    track3 = pygame.image.load('images/track3.jpg')
    track3BW = Image.open('images/track3BW.jpg').convert('1')

    if k == 1:
        return track1, np.array(track1BW)
    elif k == 2:
        return track2, np.array(track2BW)
    elif k == 3:
        return track3, np.array(track3BW)
    else:
        print('Wrong track!')
        exit()


tracks_params = {
    1: {'x': 580, 'y': 300, 'angle': 0},
    2: {},
    3: {}
}

'''
TODO:
1. Neural net visualization, augment the track

'''
if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Evolutionary cars")

    background, TRACK_MAP = load_racing_tracks(k=int(input('Which track?: ')))
    window_height, window_width = TRACK_MAP.shape

    print(f'WINDOW SIZE: {window_width} x {window_height}')

    win = pygame.display.set_mode((window_width, window_height))
    # font = pygame.font.Font('../font/font.ttf', 100)

    my_car = car.Car(x=580, y=300, angle=0, n_sensors=5, sprite_path='images/car.png')
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        win.blit(background, dest=(0, 0))

        my_car.update_position(TRACK_MAP)
        my_car.draw(win)

        pygame.display.update()
        time.sleep(0.1)
