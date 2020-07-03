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

* Track 1 - easy
* Track 2 - hard
* Track 3 - very hard - waiting for introduction


#### Miscellaneous:
1. Program saves the best cars it has seen.
2. Program can save all sensors measurements during the laps.


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
1. <h4>Parent Selection</h4>

	From the current population, we choose *parents* such that the probability of choosing a parent *p* is equal to its fitness value *f(x)*, where:

	 <img src="https://render.githubusercontent.com/render/math?math=f(x_i) = \frac{F(x_i) - F_{min}}{\sum_{j=1}^{n}(F(x_j)-F_{min})}">

	 *F(x) is an objective function value*.

2. <h4>Crossovers</h4>

	Parents chosen in the previous step pair up to produce offspring.

	* <h6>Simple crossover</h6>

		Each pair produces two children.
		We draw *n* such pairs, where *n* is a population size <br> and do as follows:

		1. Iterate through genotype.
		2. With *50%* probability, *child_1* will get a gene from *parent_1*, then *child_2* gets a gene from *parent_2*.
		3. Otherwise *child_1* will get a gene from *parent_2* and *child_2* from *parent_1*.

	* <h6>Neural networks crossover</h6>

		Each pair produces one child.
		We draw *n* such pairs, where *n* is a population size.

		We have two parent neural networks *N1* and *N2*. The goal is to produce an offspring network *N3*.
		We do as follows:
		1. For each car we record the sensors measurements and car's decision at every frame. 
		2. From each parent we draw *k* samples of those measurements and corresponding predicts.
		3. We feed *2k* samples to the child network N3.
		4. We hope that N3 learned something and drives not worst than it's parents.


3. <h4>Mutations</h4>
    
   * <h6>Simple mutation</h6>
    
        Add a Gaussian Noise to the network.
        
   * <h6>ES mutation</h6>
    
        Use ES(λ + μ) algorithm for mutation. 
        This is quite powerful algorithm, 
        which gives us a possibility to run it alone without a crossover.


## Screenshots
* Sensor

	![GIF](/videos/scanners.gif)

* Track 1

	![GIF](/videos/v1.gif)

* Track2

	![GIF](/videos/v2.gif)

* Track 3

	![image](/images/track3/track3.jpg)

## Technologies
* Python - version 3.7.3

## Libraries
* numpy
* pygame
* Image
* pytorch
* json


## Results
* Cars have no problems with learning how to deal with the first track.
* For now they cannot finish 2nd track, because they speed up to much near the end.
* Neural network crossover works quite well, but it is slow.
* Simple and fast ES is the best way to go here.

## Status
Project is: _in_the_middle_

## Credits
Created by [@TheFebrin](https://github.com/TheFebrin) <br>
Racing tracks <a href="http://www.freepik.com">Designed by Freepik</a>
