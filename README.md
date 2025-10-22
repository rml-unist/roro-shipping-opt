<p align="center">
<h1 align="center">Roll-on Roll-off Shipping Optimization</h1>
</p>

This repository is the final source code submitted by FAST for Optimization Grand Challenge 2025, organized by LG CNS.

<p align="center">
<img src="./image/image.png" width="400" height="280">
</p>
Reference : https://www.news1.kr/economy/trend/5312046 ⓒ News1 윤일지 기자

---

## Contact

- Jungeun Lee: jungeunlee14@gmail.com
- Changju Kim: kcj4746@unist.ac.kr

---

## Problem Overview

This project addresses the Roll-on Roll-off (RoRo) shipping optimization problem, where vehicles are loaded and unloaded inside a vessel structured as a network of nodes and edges.  
- Fixed port sequence: The vessel visits ports in a predefined order (Port 1 $\rightarrow$ Port 2 $\rightarrow$ $\cdots$).
- Vehicle operations: At each port, certain vehicles must be loaded (onboarded) or unloaded (offboarded).
- Ship layout: The ship's internal structure is modeled as a graph network, where each node represents a parking position.
- Blocking constraints: A vehicle can move only if its path is not blocked by others; loading and unloading must occur through feasible, unblocked paths.
- Repositioning: Vehicles not scheduled for loading or unloading can be relocated to the other nodes.
- Temporary unloading: Vehicles not due for unloading may be unloaded and reloaded if they block another vehicle's route.

#### Goal  
The goal of this problem is to minimize the total cost incurred during loading, unloading, and relocation of vehicles within the ship network.  
 
$$\min \sum_i \left(\textrm{fixed cost}_i + \textrm{variable cost}_i\right)$$

- Fixed cost: Charged once for each loading or unloading operation (i.e., when a vehicle is oved onto or off the ship).
- Variable cost: Proportional to the path length, which is measured by the number of edges traversed in the network.

---

## Preliminary Setup

### Set up environment
Create conda environment in the repository.
```bash
conda env create -f ogc2025.yaml
conda activate ogc2025
```

### Install dependencies
```bash
pip install pybind11
```

### Generate a C++ backend module
```bash
cd ./c_backend
c++ -O3 -Wall -shared -std=c++17 -fPIC `python -m pybind11 --includes` engine.cpp -o engine`python3-config --extension-suffix`
```

### Move the generated ``.so`` file to an external directory
For example, if the generated file is named ``engine.so``,
```bash
cp engine.so ../
cd ../
```

---

## Run
```bash
python3 myalgorithm.py <PROBLEM_NAME> <PROBLEM_FILE_PATH> <TIMELIMIT>
```

---

## Video
https://youtu.be/k6kmN5ykTqw

