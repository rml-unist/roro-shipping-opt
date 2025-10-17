<p align="center">
<h1 align="center">Port Optimization</h1>
</p>

This repository is the final source code submitted by FAST for Optimization Grand Challenge 2025, organized by LG CNS.

<p align="center">
<img src="./imgage/image.png" width="300" height="180">
</p>
Reference : https://www.news1.kr/economy/trend/5312046 ⓒ News1 윤일지 기자

---

## Contact

- Jungeun Lee: jungeunlee14@gmail.com
- Changju Kim: kcj4746@unist.ac.kr

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

