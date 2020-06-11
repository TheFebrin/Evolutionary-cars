# Evolutionary-cars
> Cars learn how to drive using evolutionary algorithms and neural networks.

## Table of contents
* [General info](#general-info)
* [Neural network](#neural-network)
* [Objective function](#objective-function)
* [Genetic algorithm](#genetic-algorithm)
* [Screenshots](#screenshots)
* [Technologies](#technologies)
* [Libraries](#libraries)
* [Results](#results)
* [Status](#status)

## General info

#### Cars:
1. Each car has **k** sensors.
2. Car's position is calculated using its current velocity and angle.
3. Each sensor detects the distance from the car to the race track barrier.
3. Those measurements are inputs of the neural network that predicts the action.
4. The neural network learns using an evolutionary algorithm.
5. Car's velocity and angle are changed according to the network's output.

#### Tracks:
TODO

#### Miscellaneous:
1. Program saves best cars it has seen.
TODO


## Neural network
I used a simple network with 2 hidden layers.
![Network](/images/network.jpg)

* Input Layer:
	(**k** sensors measurements, actual speed, actual angle)

* Output Layer:
	(angle change, acceleration change)

## Objective function
As it's quite hard to define an objective function I created checkpoints.

* When a car reaches a checkpoint it earns points.
* It is punished for the distance to the next checkpoint.


## Genetic algorithm
1. Parent Selection

	From the current population, we choose *parents* such that the probability of choosing a parent *p* is equal to its fitness value *f(x)*, where:

	 <img src="https://render.githubusercontent.com/render/math?math=f(x_i) = \frac{F(x_i) - F_{min}}{\sum_{j=1}^{n}(F(x_j)-F_{min})}">

	 *F(x) is an objective function value*.

2. Crossover

	Parents chosen in the previous step pair up to produce offspring.
	Each pair produces one child.
	Produced children make up half of the next generation.
	The remaining half are their parents.

	We have two parent neural networks *N1* and *N2*. The goal is to produce an offspring network *N3*.
	We do as follows:
	1. Generate 2k random genotypes.
	2. Feed k samples to each parent network.
	3. We obtain 2k outputs and treat it like a training dataset.
	3. We train N3 using this data.


3. Mutation

TODO


## Screenshots
* Sensor
	![Example screenshot](/videos/scanners.gif)

* TODO

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
