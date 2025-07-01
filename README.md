# PDM4AR Highway Driving Exercise

This repository contains the implementation for Exercise 12 of the Planning and Decision Making for Autonomous Robots (PDM4AR) course. The exercise focuses on autonomous highway driving scenarios with lane changing, collision avoidance, and goal-oriented navigation.

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
│       ├── agent.py              # Main agent implementation
│       ├── motion_primitives.py  # Vehicle motion models
│       ├── graph.py              # Path planning graph structures
│       └── dijkstra.py           # Dijkstra search algorithm
├── exercises_def/
│   └── ex12/
│       ├── ex12.py               # Exercise configuration
│       ├── sim_context.py        # Simulation setup
│       └── perf_metrics.py       # Performance evaluation
└── main.py                       # Entry point
```

## Performance Evaluation

The exercise evaluates agent performance across multiple scenarios with increasing difficulty:

### Scenario 1: Basic Highway Navigation

| Scenario 1.1 | Scenario 1.2 |
|:----------:|:----------:|
| ![1.1](videos/Evaluation-Final24-1-scenario1-EpisodeVisualisation-figure1-Animation.gif) | ![1.2](videos/Evaluation-Final24-1-scenario2-EpisodeVisualisation-figure1-Animation.gif) |

*Description: Basic lane following and lane changing in light traffic conditions.*

### Scenario 2: Complex Traffic Situations

| Scenario 2.1 | Scenario 2.2 |
|:----------:|:----------:|
| ![2.1](videos/Evaluation-Final24-2-scenario1-EpisodeVisualisation-figure1-Animation.gif) | ![2.2](videos/Evaluation-Final24-2-scenario2-EpisodeVisualisation-figure1-Animation.gif) |

*Description: Advanced scenarios with dense traffic, requiring strategic lane changes.*

### Scenario 3: Emergency Situations

| Scenario 3.1 |
|:----------:|
| ![3.1](videos/Evaluation-Final24-3-scenario1-EpisodeVisualisation-figure1-Animation.gif) |

*Description: Difficult scenario with traffic jam.*

## Evaluation Metrics

The agent performance is assessed using the following criteria:

- **Safety**: Collision avoidance and safe following distances
- **Efficiency**: Time to goal and smooth trajectory execution  
- **Comfort**: Minimal jerk and acceleration variations
- **Goal Achievement**: Successful navigation to target destinations
- **Traffic Compliance**: Adherence to traffic rules and lane discipline

## References

- [PDM4AR Course Materials](https://pdm4ar.github.io/exercises/12-highway_driving.html)
