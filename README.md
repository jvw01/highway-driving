# PDM4AR Highway Driving Exercise

This repository contains the implementation for the final exercise of the Planning and Decision Making for Autonomous Robots (PDM4AR) course.
The exercise focuses on autonomous highway driving scenarios with lane changing, collision avoidance, and goal-oriented navigation.

## Overview

The highway driving exercise challenges students to implement a complete autonomous driving agent capable of:
- **Lane keeping and following**: Maintaining position within lane boundaries
- **Dynamic obstacle avoidance**: Responding to other vehicles and dynamic obstacles
- **Strategic lane changing**: Making intelligent decisions for overtaking and goal achievement
- **Velocity control**: Adaptive speed management based on traffic conditions
- **Goal-oriented navigation**: Efficient path planning to reach designated targets

## Getting Started

### Prerequisites

- Docker (for devcontainer support)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jvw01/highway-driving.git
   cd highway-driving
   ```

2. Set up the development environment using the provided devcontainer or install dependencies manually.

### Running the Exercise

Execute Exercise 12 using the following command:

```bash
python src/pdm4ar/main.py --exercise 12
```

## Project Structure

```
src/pdm4ar/
├── exercises/
│   └── ex12/
│       ├── agent.py  # agent coordinator (state machine)
│       ├── __init__.py
│       │
│       ├── controllers/
│       │   ├── __init__.py
│       │   ├── velocity_controller.py  # controller for safe following
│       │   └── steering_controller.py  # controller for lane keeping
│       │
│       ├── planning/
│       │   ├── __init__.py
│       │   ├── motion_primitives.py  # motion primitive implementation
│       │   ├── motion_primitive_manager.py  # motion primitive generation
│       │   ├── lane_change.py  # calculate time horizon and ddelta for lane change maneuver
│       │   ├── collision_checker.py  # collision detection and avoidance
│       │   ├── graph.py  # graph generation for shortest path search
│       │   └── dijkstra.py  # Dijkstra's algorithm
│       │
│       └── utils/
│           ├── __init__.py
│           └── visualization.py  # Debug plotting and visualization
|
└── main.py  # Entry point
```

## Performance Evaluation

The exercise evaluates agent performance across multiple scenarios with increasing difficulty.
This implementation manages to pass both the public and private test cases in the course "Planning and Decision Making for Autonomous Robots".

### Scenario 1: Basic Highway Navigation

Basic lane following and lane changing in light traffic conditions.

| Scenario 1.1 | Scenario 1.2 |
|:----------:|:----------:|
| ![1.1](videos/Evaluation-Final24-1-scenario1-EpisodeVisualisation-figure1-Animation.gif) | ![1.2](videos/Evaluation-Final24-1-scenario2-EpisodeVisualisation-figure1-Animation.gif) |

### Scenario 2: Complex Traffic Situations

Advanced scenarios with dense traffic, requiring strategic lane changes.

| Scenario 2.1 | Scenario 2.2 |
|:----------:|:----------:|
| ![2.1](videos/Evaluation-Final24-2-scenario1-EpisodeVisualisation-figure1-Animation.gif) | ![2.2](videos/Evaluation-Final24-2-scenario2-EpisodeVisualisation-figure1-Animation.gif) |

### Scenario 3: Emergency Situations

Difficult scenario. Lane changing in traffic jam.

| Scenario 3 |
|:----------:|
| ![3](videos/Evaluation-Final24-3-scenario1-EpisodeVisualisation-figure1-Animation.gif) |

## Evaluation Metrics

The agent performance is assessed using the following criteria:

- **Safety**: Collision avoidance and safe following distances
- **Efficiency**: Time to goal and smooth trajectory execution  
- **Comfort**: Minimal jerk and acceleration variations
- **Goal Achievement**: Successful navigation to target destinations
- **Traffic Compliance**: Adherence to traffic rules and lane discipline

## References

- [PDM4AR Course Materials](https://pdm4ar.github.io/exercises/12-highway_driving.html)
