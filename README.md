# Siesta
State-Inferred Electric Servo Torque Approximation

## Description
This repo comprises work related to my master thesis, in which I aim to learn the model dynamics of a Dynamixel XM430-W210 servo-motor using supervised learning. 

It contains:
- Tools for gathering experimental data
- Scripts for preprocessing, training, and evaluating the neural network models

## Filesystem
A brief description of the relevant directories

/config/:
- Experiment setup parameters

/data:
- /measurements/: Datasets from measurement experiments land here
- /measurements_quail_gazebo/: Datasets from simulated measurement experiments in gazebo land here
- /flight_data/: Position and setpoint data from real flights
- /input_signals/: Generated input signals 
- /models/: Trained networks
- /training/: Final training datasets
- /evaluation/: Final evaluation datasets of different input signal types

/include/, /launch/ & /src/:
- C++ files to run real and simulated(gazebo) measurements using ROS

/scripts/:
- Python scripts for training purposes 

## Instructions
Clone this repo into you catkin workspace's src/ directory

### Data gathering
The 'omav_hovery' drivers are used to interface the dynamixel motor. Clone the repo into your catkin workspace using
~~~
git clone --recurse-submodules https://github.com/ethz-asl/omav_hovery
~~~
and run all its necessary install scripts.


Then, build the ros nodes with 
~~~
catkin build siesta
~~~
After connecting and turn on the motor controller, set the experiment's input signal and load_id in the config file and launch the exeperiment:
~~~
roslaunch siesta data_collection.launch
~~~

This program generates csv files with the captured motor data. The filenames consist of the date and time, the input signal type, and the used load. 

### Model training 
The python script for training a network using experimental data is located in the /scripts/ directory.

*siesta_mlp.py*:
- Read the csv dataset
- Compute velocity & acceleration from positions
- Prepare network data (inputs & target) with a given position history length
- Train model
- Evaluate model

### Model evaluation
The trained models can be integrated into gazebo and measured using:

roslaunch siesta siesta_evaluation_gazebo.launch

The measured dataset can be compared to true values with the *compare_signals.py* scripts.

More efficiently, the following script:

*infer_model.py*:
- Runs a simulation of the trained networks response to a specific input signal
- Compares it to the PD model and true measurements (in ../evaluation/))
