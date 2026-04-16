# Aero Hands - Interactive CFD Wind Tunnel

This project is a real-time fluid dynamics laboratory that utilises hand tracking for spatial manipulation of aerofoils. It is built to bridge the gap between high-performance physics simulations and intuitive human-computer interaction, allowing for the rapid testing and logging of aerodynamic profiles.

## Core Architecture
The system is built on a modular stack designed for low-latency feedback:
* **Physics Engine:** A 2D Navier-Stokes solver implemented in **Taichi Lang**, capable of running accelerated fluid simulations on the GPU/CPU.
* **Vision Engine:** **MediaPipe** skeletal tracking for hand landmark detection.
* **Geometry Engine:** A custom transformation handler using **float32 precision** to prevent geometric degradation during high-frequency rotation and scaling.

## Key Functionality
The simulation allows for three primary modes of interaction:
1.  **Custom Geometry:** The right hand acts as a drawing tool. Pinching the index and thumb allows the user to trace custom shapes directly into the airstream.
2.  **Spatial Manipulation:** The left hand acts as a 3D controller. Open palm movement translates the object in the X/Y plane, while moving the hand closer or further from the camera scales the object (Z-axis depth).
3.  **Data Analysis:** The system calculates real-time coefficients for Lift ($C_l$), Drag ($C_d$), and the $L/D$ Ratio. These are derived from the integrated pressure and velocity fields around the obstacle.



## Mathematical Grounding
The simulation normalises forces to provide standard coefficients. The lift force is calculated as:

$$L = C_l \cdot \frac{1}{2} \rho v^2 S$$

Where:
* $C_l$ is the Lift Coefficient.
* $\rho$ is the fluid density.
* $v$ is the wind velocity.
* $S$ is the chord length (surface area in 2D).


## Setup and Installation
1.  Clone the repository.
2.  Install dependencies: `pip install -r requirements.txt` (requires `taichi`, `opencv-python`, `mediapipe`, and `numpy`).
3.  Execute: `python main.py`

## User Controls
* **[S]** Toggle simulation (Start/Pause).
* **[W / S]** Increment/Decrement Angle of Attack (AoA).
* **[L]** Log current physics state to a timestamped CSV file.
* **[1 - 5]** Load NACA 4-digit aerofoil presets (Symmetric, Cambered, Thin, Flat Plate, Heavy Lift).
* **[C]** Clear current geometry and reset fluid state.
* **[Q]** Terminate application.

## Future Development
The project is structured to eventually support Reinforcement Learning (RL) environments. By utilising the logged performance data as a reward signal, an agent can be trained to optimise aerofoil geometry for specific aerodynamic targets. Secondary goals include the implementation of Bernoulli-based pressure/velocity heatmaps and live performance polar plotting.