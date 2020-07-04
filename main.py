from PIL import Image
import pygame
import numpy as np
import time
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from cars import car
from genetic_algorithms import genetic_algorithm
matplotlib.use("Agg")

TRACKS_PARAMS = {
    1: {'x': 570, 'y': 360, 'angle': 0},
    2: {'x': 430, 'y': 840, 'angle': -90},
    3: {}
}

CHECKPOINTS_POSITIONS = {
    1: [(530, 600), (150, 600), (150, 160), (530, 160)],
    2: [(120, 750), (75, 225), (290, 70), (500, 220), (610, 450), (725, 630), (660, 825)],
}

DRAW_POSITIONS = {
    1: {'plot': (700, 350)},
    2: {'plot': (850, 400)}
}

BEST_CARS = {}
N_BEST_CARS: int = 50
actual_best: int = 0


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
        checkpoints = [
            Image.open(f'images/track1/checkpoint{i}.jpg').convert('1')
            for i in range(4)
        ]
    if k == 2:
        checkpoints = [
            Image.open(f'images/track2/checkpoint{i}.jpg').convert('1')
            for i in range(7)
        ]

    return list(map(np.array, checkpoints))


def draw_checkpoints(win):
    for i, t in enumerate(CHECKPOINTS_POSITIONS[track]):
        x, y = t
        pygame.draw.circle(win, (255, 255, 0), (x, y), 8)
        id = font.render(f'{i}', True, (255, 255, 0))
        win.blit(id, dest=(x - 6, y + 12))


def draw_actual_best(win):
    ab = actual_best / car.CHECKPOINT_AWARD
    ab_text = font.render(f'Current best: {ab:.5f}', True, (0, 0, 255))
    win.blit(ab_text, dest=(600, 100))


def draw_info(win):
    space = '(SPACE) - kill generation'
    font = pygame.font.SysFont('arial', 20)
    space_text = font.render(space, True, (0, 255, 100))
    win.blit(space_text, dest=(700, 150))

    L = '(L) - load best cars'
    l_text = font.render(L, True, (0, 255, 100))
    win.blit(l_text, dest=(700, 200))


fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)
canvas = agg.FigureCanvasAgg(fig)
objective_values_plot = None


def draw_objective_values(win, model, draw=False):
    global objective_values_plot

    if len(model.cost_history) == 0:
        return

    if draw:
        win.blit(objective_values_plot, dest=DRAW_POSITIONS[track]['plot'])
        return

    objective_values_plot = np.array(model.cost_history) / car.CHECKPOINT_AWARD
    ax.plot(objective_values_plot[:, 0], color='green')
    ax.plot(objective_values_plot[:, 1], color='orange')
    ax.plot(objective_values_plot[:, 2], color='red')
    plt.legend(['Min', 'Mean', 'Max'], loc='upper right')
    plt.title('Objective values')
    plt.xlabel('#generation')
    plt.ylabel('value (#laps)')
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()
    objective_values_plot = pygame.image.fromstring(raw_data, size, "RGB")

    win.blit(objective_values_plot, dest=DRAW_POSITIONS[track]['plot'])


def load_best_cars(model, cars_list, s, track=1):
    global BEST_CARS
    path = f'cars/track{track}_best_cars_{s}sensors.json'
    BEST_CARS = json.load(open(path, encoding='utf-8'))
    BEST_CARS = {float(k): v for k, v in BEST_CARS.items()}
    for i, t in enumerate(BEST_CARS.items()):
        if i >= len(cars_list):
            break
        model.population[i] = np.array(t[1])
        cars_list[i].genotype = np.array(t[1])
        # cars_list[i].objective_value = t[0]


def update_fps(win, clock, time_start):
    fps = f'FPS: {int(clock.get_fps())}  |  Time: {time.time() - time_start:.2f}'
    fps_text = font.render(fps, 1, pygame.Color("blue"))
    win.blit(fps_text, dest=(600, 50))


def draw_hall_of_fame(win):
    best_cars = list(BEST_CARS.keys())
    best_cars.sort(reverse=True)
    for i, score in enumerate(best_cars):
        car_score = font.render(
            f'{i + 1}. {score / car.CHECKPOINT_AWARD:.5f}',
            True,
            (0, 0, 255)
        )
        win.blit(car_score, dest=(1050, 50*(i + 1) + 10))
        if i >= 4:
            break


def update_cars_population(genetic_model, cars_list, win, n_gen):
    global actual_best
    objective_values = np.array(
        [c.objective_value for c in cars_list],
        dtype=np.float64
    )
    training_data = np.array([
        np.array(c.training_data) for c in cars_list
    ])

    print(f'Learning samples: {training_data[0].shape}')

    actual_best = objective_values.max()
    print(f'Best objective value: {actual_best}')

    genetic_model.cost = objective_values
    genetic_model.training_data = training_data
    new_genotypes = genetic_model.select_new_population(
        n_gen,
        crossover=CROSSOVER_TYPE
    )

    for i, c in enumerate(cars_list):
        sprite_path = 'images/cars/car.png'
        # if c.id in ids:
        #     sprite_path = 'images/cars/car2.png'

        BEST_CARS[c.objective_value] = c.genotype.tolist()

        c.sprite = pygame.image.load(sprite_path)
        c.sprite = pygame.transform.scale(c.sprite, c.sprite_size)
        c.x = TRACKS_PARAMS[track]['x']
        c.y = TRACKS_PARAMS[track]['y']
        c.angle = TRACKS_PARAMS[track]['angle']
        c.dead = False
        c.velocity = 0
        c.time_start = time.time()
        c.genotype = new_genotypes[i]
        c.next_checkpoint = 0
        c.objective_value = 1
        c.training_data = []



if __name__ == '__main__':
    pygame.init()
    pygame.display.set_caption("Evolutionary cars")

    '''
    Print information and input track number
    '''
    print('\n' + '-' * 50 + '\n')
    print('Track 1: Easy\nTrack 2: Medium\nTrack 3: Hard')
    print('(Tracks 1-2 are for training, Track 3 is for testing.)')
    print('\n' + '-' * 50 + '\n')
    track = int(input('Which track?: (1 / 2 / 3): '))
    n_cars = int(input('\nNumber of cars in population (even): '))
    assert(n_cars % 2 == 0)  # for parents selection sake

    '''
    Load track map and checkpoints
    '''
    background, TRACK_MAP = load_racing_tracks(k=track)
    CHECKPOINTS_MAPS = load_checkpoints(k=track)
    window_height, window_width = TRACK_MAP.shape

    print(f'WINDOW SIZE: {window_width} x {window_height}')

    '''
    Set up pygame entities
    '''
    win = pygame.display.set_mode((window_width, window_height))
    clock = pygame.time.Clock()
    font = pygame.font.Font('font/font.ttf', 40)
    gen = font.render('Generation: 1', True, (0, 0, 255))
    top = font.render('Most checkpoints:', True, (0, 0, 255))

    '''
    Initialize cars population
    '''
    n_sensors = 7
    cars_list = [car.Car(
        id=i,
        x=TRACKS_PARAMS[track]['x'],
        y=TRACKS_PARAMS[track]['y'],
        angle=TRACKS_PARAMS[track]['angle'],
        n_sensors=n_sensors,
        sprite_path='images/cars/car.png')
        for i in range(n_cars)
    ]

    '''
    Initialize genetic model
    '''
    EVOLVE = True  # Population is freezed if False
    CROSSOVER_TYPE = 1  # neural network crossover
    # CROSSOVER_TYPE = 2  # random crossover
    n_parameters = cars_list[0].n_parameters
    genetic_model = genetic_algorithm.GA(
        n_sensors=n_sensors,
        population_size=n_cars,
        chromosome_len=n_parameters, K=1,
        evolve=EVOLVE
    )

    '''
    Load top cars from *track* into population
    '''
    LOAD_BEST_CARS = True
    if LOAD_BEST_CARS:
        load_best_cars(genetic_model, cars_list, n_sensors, track=track)

    '''
     ------ MAIN LOOP --------------  MAIN LOOP  -----------
    '''
    running = True
    n_gen = 0
    time_start = time.time()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE] or time.time() - time_start >= 60:  # SPACE skips to next generation
            for c in cars_list:
                nc = CHECKPOINTS_POSITIONS[track][c.next_checkpoint]
                c.objective_value -= (  # no sqrt
                    (c.x - nc[0]) ** 2 + (c.y - nc[1]) ** 2
                )
                c.dead = True
        if keys[pygame.K_l]:  # l loads hall of fame cars
            n_gen = 0
            load_best_cars(genetic_model, cars_list, n_sensors, track=track)
            for c in cars_list:
                c.dead = True

        win.blit(background, dest=(0, 0))

        for c in cars_list:
            c.update_position(TRACK_MAP, CHECKPOINTS_MAPS, CHECKPOINTS_POSITIONS[track])
            c.predict_move()
            c.draw(win)

        win.blit(gen, (600, 0))   # draw generation number
        win.blit(top, (1050, 0))  # draw top cars (hall of fame)

        draw_actual_best(win)     # draw score of the best individual
        draw_hall_of_fame(win)
        draw_checkpoints(win)
        draw_info(win)
        draw_objective_values(win, genetic_model, draw=True)
        update_fps(win, clock, time_start)

        dead_cars = sum([c.dead for c in cars_list])
        if dead_cars == n_cars:
            print('\n' + '-' * 60 + '\n')
            n_gen += 1
            print(f'Generation: {n_gen}')
            gen = font.render(f'Generation: {n_gen}', True, (0, 0, 255))
            '''
            #  Saving sample measurements that cars can see
            out = f'cars/training _data_track1.npy'
            all_training_data = []
            for c in cars_list:
                all_training_data += c.training_data
            all_training_data = np.array(all_training_data)
            with open(out, 'w') as outfile:
                np.save(out, all_training_data)
            '''
            update_cars_population(genetic_model, cars_list, win, n_gen)
            draw_objective_values(win, genetic_model, draw=False)
            time_start = time.time()
            while len(BEST_CARS) > N_BEST_CARS:
                BEST_CARS.pop(min(BEST_CARS))

            if LOAD_BEST_CARS:
                out = f'cars/track{track}_best_cars_{n_sensors}sensors.json'
                with open(out, 'w') as outfile:
                    json.dump(BEST_CARS, outfile, ensure_ascii=False)

        pygame.display.update()
        clock.tick(60)

    genetic_model.plot_cost()
    genetic_model.plot_sigmas(genetic_model.best_sigmas_history, mode='best')
    genetic_model.plot_sigmas(genetic_model.sigmas_history, mode='all')
