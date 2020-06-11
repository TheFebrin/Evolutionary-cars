# Evolutionary-cars
> Cars learn how to drive using evolutionary algorithms and neural networks.

## Table of contents
* [General info](#general-info)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Results](#results)
* [Status](#status)

## General info
TODO

1. Each car has **k** sensors.
2. Each sensor detects the distance from the car to the race track barrier.
3. Those measurements are inputs of the neural network that predicts the action.
4. The neural network learns using an evolutionary algorithm.


## Actions
Variety of inputs / Input possibilities:
	1. (**k** sensors measurements)
	2. (**k** sensors measurements, actual speed, actual angle)

Outputs:
	1. (angle) - speed is fixed
	2. (angle change, acceleration change)


## Objective function
- Yellow checkpoints


## Genetic algorithm
1. Parent Selection

	From the current population, we choose *parents* such that the probability of choosing a parent $p$ is equal to its fitness value f(x)

2. Crossover

	We have two parent neural networks N1 and N2. The goal is to produce an offspring network N3.
	We do as follows:
	1. Generate 2k random genotypes.
	2. Feed k samples to each parent network.
	3. We obtain 2k outputs and treat it like a training dataset.
	3. We train N3 using this data.


3. Mutation
TODO


## Screenshots
![Example screenshot](/videos/scanners.gif)

## Technologies
* Python - version 3.7.3

## Libraries
* numpy
* pygame
* Image
* TODO

## Results
* TODO

## Status
Project is: _just_started_

## Credits
Created by [@TheFebrin](https://github.com/TheFebrin) <br>
Racing tracks <a href="http://www.freepik.com">Designed by Freepik</a>
