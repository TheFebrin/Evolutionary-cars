from PIL import Image
import pygame
import numpy as np
import time
import json

from cars import car
from genetic_algorithms import ES

TRACKS_PARAMS = {
    1: {'x': 570, 'y': 360, 'angle': 0},
    2: {},
    3: {}
}

CHECKPOINT_AWARD = {
    1: [(530, 600), (150, 600), (150, 160), (530, 160)]

}

BEST_CARS = json.load(open('cars/best_cars.json', encoding='utf-8'))
BEST_CARS = {float(k): v for k, v in BEST_CARS.items()}


def load_racing_tracks(k):
    track1 = pygame.image.load('images/track1/track1.jpg')
    track1BW = Image.open('images/track1/track1BW.jpg').convert('1')
    track2 = pygame.image.load('images/track2/track2.jpg')
    track2BW = Image.open('images/track2/track2BW.jpg').convert('1')
    track3 = pygame.image.load('images/track3/track3.jpg')
    track3BW = Image.open('images/track3/track3BW.jpg').convert('1')

    if k == 1:
        return track1, np.array(track1BW)
    elif k == 2:
        return track2, np.array(track2BW)
    elif k == 3:
        return track3, np.array(track3BW)
    else:
        print('Wrong track!')
        exit()


def load_checkpoints(k):
    if k == 1:
        c0 = Image.open('images/track1/checkpoint0.jpg').convert('1')
        c1 = Image.open('images/track1/checkpoint1.jpg').convert('1')
        c2 = Image.open('images/track1/checkpoint2.jpg').convert('1')
        c3 = Image.open('images/track1/checkpoint3.jpg').convert('1')

        return list(map(np.array, [c0, c1, c2, c3]))


def draw_checkpoints(win):
    for i, t in enumerate(CHECKPOINT_AWARD[track]):
        x, y = t
        pygame.draw.circle(win, (255, 255, 0), (x, y), 8)
        id = font2.render(f'{i}', True, (255, 255, 0))
        win.blit(id, dest=(x - 6, y + 12))


def load_best_cars():
    pass


'''
TODO:
1. Neural net visualization, augment the track
2. space desc and additional keys
3. ES desc
4. save weights
5. top cars, best weights
6. krzyzowanie sieci
7. wykres kiedy skrzyzowane sieci daly cos lepszego
8. track 3 to test track
'''

if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Evolutionary cars")

    # track = int(input('Which track?: (1 / 2 / 3)'))
    track = 1
    background, TRACK_MAP = load_racing_tracks(k=track)
    CHECKPOINTS_MAPS = load_checkpoints(k=track)
    window_height, window_width = TRACK_MAP.shape

    print(f'WINDOW SIZE: {window_width} x {window_height}')

    win = pygame.display.set_mode((window_width, window_height))
    font = pygame.font.Font('font/font.ttf', 40)
    font2 = pygame.font.Font(pygame.font.get_default_font(), 24)

    gen = font.render('Generation: 1', True, (0, 0, 255))
    top = font.render('Top cars (#laps):', True, (0, 0, 255))

    n_cars = 20
    cars = [car.Car(
        id=i,
        x=TRACKS_PARAMS[track]['x'],
        y=TRACKS_PARAMS[track]['y'],
        n_sensors=5,
        sprite_path='images/cars/car.png')
        for i in range(n_cars)]

    n_parameters = cars[0].n_parameters
    genetic_model = ES.ES(mu=n_cars, lambda_=5,
                          chromosome_len=n_parameters, K=1)

    running = True
    n_gen = 0
    while running:  # ------ MAIN LOOP --------------  MAIN LOOP  -----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:
            for c in cars:
                nc = CHECKPOINT_AWARD[track][c.next_checkpoint]
                c.objective_value -= np.sqrt(
                    (c.x - nc[0]) ** 2 + (c.y - nc[1]) ** 2
                )
                c.dead = True

        win.blit(background, dest=(0, 0))

        for c in cars:
            c.update_position(TRACK_MAP, CHECKPOINTS_MAPS, CHECKPOINT_AWARD[track])
            c.predict_move()
            c.draw(win)

        win.blit(gen, (600, 0))
        win.blit(top, (1150, 0))

        best_cars_iter = iter(BEST_CARS.items())
        for i in range(1, 11):
            act = next(best_cars_iter, False)
            car_score = font2.render(f'{i - 1}. {act[0] // cars.CHECKPOINT_AWARD}', True, (0, 0, 255)) if act \
                            else font2.render(f'{i - 1}. ???', True, (0, 0, 255))
            win.blit(car_score, (1150, 30*i + 10))
        draw_checkpoints(win)

        dead_cars = sum([c.dead for c in cars])
        if dead_cars == n_cars:
            n_gen += 1
            gen = font.render(f'Generation: {n_gen}', True, (0, 0, 255))

            objective_values = np.array(
                [c.objective_value for c in cars],
                dtype=np.float64
            )
            print(f'Objective values gen {n_gen - 1}:\n{objective_values}\n')
            genetic_model.cost = objective_values
            new_genotypes, ids = genetic_model.select_new_population()
            ids = set(ids)
            print(f'Best cars: {ids}')
            for i, c in enumerate(cars):
                sprite_path = 'images/cars/car.png'
                if c.id in ids:
                    sprite_path = 'images/cars/car2.png'

                BEST_CARS[c.objective_value] = c.genotype.tolist()

                c.sprite = pygame.image.load(sprite_path)
                c.sprite = pygame.transform.scale(c.sprite, c.sprite_size)
                c.x = TRACKS_PARAMS[track]['x']
                c.y = TRACKS_PARAMS[track]['y']
                c.dead = False
                c.velocity = 0
                c.angle = 0
                c.time_start = time.time()
                c.genotype = new_genotypes[i]
                c.next_checkpoint = 0
                c.objective_value = 1
            while len(BEST_CARS) > 10:
                BEST_CARS.pop(min(BEST_CARS))

            with open('cars/best_cars.json', 'w') as outfile:
                json.dump(BEST_CARS, outfile, ensure_ascii=False)

        pygame.display.update()

    genetic_model.plot_cost()
    genetic_model.plot_sigmas(genetic_model.best_sigmas_history, mode='best')
    genetic_model.plot_sigmas(genetic_model.sigmas_history, mode='all')
