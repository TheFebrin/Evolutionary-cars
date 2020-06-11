from PIL import Image
import pygame
import numpy as np
import time
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
from cars import car
from genetic_algorithms import ES, genetic_algorithm
matplotlib.use("Agg")

TRACKS_PARAMS = {
    1: {'x': 570, 'y': 360, 'angle': 0},
    2: {},
    3: {}
}

CHECKPOINT_AWARD = {
    1: [(530, 600), (150, 600), (150, 160), (530, 160)]

}

BEST_CARS = json.load(open('cars/track1_best_cars.json', encoding='utf-8'))
BEST_CARS = {float(k): v for k, v in BEST_CARS.items()}
N_BEST_CARS = 5
actual_best = 0


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


fig = plt.figure(figsize=[6, 4])
ax = fig.add_subplot(111)
canvas = agg.FigureCanvasAgg(fig)
objective_values_plot = None


def draw_objective_values(win, model, draw=False):
    global objective_values_plot

    if len(model.cost_history) == 0:
        return

    if draw:
        win.blit(objective_values_plot, dest=(700, 350))
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

    win.blit(objective_values_plot, dest=(700, 350))


def load_best_cars(model, cars_list):
    for i, t in enumerate(BEST_CARS.items()):
        cars_list[i].genotype = np.array(t[1])
        cars_list[i].objective_value = t[0]
        model.population[i] = np.array(t[1])


def update_fps(win, clock):
    fps = f'FPS: {int(clock.get_fps())}'
    fps_text = font.render(fps, 1, pygame.Color("blue"))
    win.blit(fps_text, dest=(600, 50))


def draw_hall_of_fame(win):
    best_cars = list(BEST_CARS.keys())
    best_cars.sort(reverse=True)
    for i, score in enumerate(best_cars):
        car_score = font.render(
            f'{i}. {score / car.CHECKPOINT_AWARD:.5f}',
            True,
            (0, 0, 255)
        )
        win.blit(car_score, dest=(1050, 50*(i + 1) + 10))


def update_cars_population(genetic_model, cars_list, win):
    global actual_best
    objective_values = np.array(
        [c.objective_value for c in cars_list],
        dtype=np.float64
    )
    actual_best = objective_values.max()
    print(f'Best objective value: {actual_best}')
    # print(f'Objective values gen {n_gen - 1}:\n{objective_values}\n')
    genetic_model.cost = objective_values
    new_genotypes, ids = genetic_model.select_new_population()
    ids = set(ids)
    print(f'Best cars: {ids}')
    for i, c in enumerate(cars_list):
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


'''
TODO:
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
    clock = pygame.time.Clock()
    font = pygame.font.Font('font/font.ttf', 40)

    gen = font.render('Generation: 1', True, (0, 0, 255))
    top = font.render('Hall of fame (#laps):', True, (0, 0, 255))

    n_cars = 20
    n_sensors = 5
    cars_list = [car.Car(
        id=i,
        x=TRACKS_PARAMS[track]['x'],
        y=TRACKS_PARAMS[track]['y'],
        n_sensors=n_sensors,
        sprite_path='images/cars/car.png')
        for i in range(n_cars)
    ]

    n_parameters = cars_list[0].n_parameters
    genetic_model = genetic_algorithm.GA(
        n_sensors=n_sensors,
        population_size=n_cars,
        chromosome_len=n_parameters, K=1
    )

    '''
    Top cars into population
    '''
    load_best_cars(genetic_model, cars_list)

    running = True
    n_gen = 0
    while running:  # ------ MAIN LOOP --------------  MAIN LOOP  -----------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break

        keys = pygame.key.get_pressed()
        if keys[pygame.K_SPACE]:  # SPACE skips to next generation
            for c in cars_list:
                nc = CHECKPOINT_AWARD[track][c.next_checkpoint]
                c.objective_value -= np.sqrt(
                    (c.x - nc[0]) ** 2 + (c.y - nc[1]) ** 2
                )
                c.dead = True

        win.blit(background, dest=(0, 0))

        for c in cars_list:
            c.update_position(TRACK_MAP, CHECKPOINTS_MAPS, CHECKPOINT_AWARD[track])
            c.predict_move()
            c.draw(win)

        win.blit(gen, (600, 0))   # draw generation number
        win.blit(top, (1050, 0))  # draw top cars (hall of fame)

        draw_actual_best(win)     # draw score of the best individual
        draw_hall_of_fame(win)
        draw_checkpoints(win)
        draw_info(win)
        draw_objective_values(win, genetic_model, draw=True)
        update_fps(win, clock)

        dead_cars = sum([c.dead for c in cars_list])
        if dead_cars == n_cars:
            print('\n' + '-' * 60 + '\n')
            n_gen += 1
            print(f'Generation: {n_gen}')
            gen = font.render(f'Generation: {n_gen}', True, (0, 0, 255))
            update_cars_population(genetic_model, cars_list, win)
            draw_objective_values(win, genetic_model, draw=False)

            while len(BEST_CARS) > N_BEST_CARS:
                BEST_CARS.pop(min(BEST_CARS))

            with open('cars/track1_best_cars.json', 'w') as outfile:
                json.dump(BEST_CARS, outfile, ensure_ascii=False)

        pygame.display.update()
        clock.tick(60)

    genetic_model.plot_cost()
    genetic_model.plot_sigmas(genetic_model.best_sigmas_history, mode='best')
    genetic_model.plot_sigmas(genetic_model.sigmas_history, mode='all')
