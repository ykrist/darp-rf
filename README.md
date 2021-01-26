# Installation
##  Requirements
Git Large-File-Storage (LFS) is required to clone the repository.

The Python code requires a working install of the `conda` package manager.  Miniconda, a minimal conda installation, 
can be found [here](https://docs.conda.io/en/latest/miniconda.html).  

A valid Gurobi license is required as well, see [here](https://www.gurobi.com/downloads/end-user-license-agreement-academic/).

## Clone 
Clone the repository with 
```
git clone --recurse-submodules https://github.com/ykrist/darp-rf.git
```

## Setup script
The setup script `setup.sh` will create a Conda environment called `darp` and install all the relevant Python packages.
If you already have a Conda environment with this name, change the `name` field in the `env.yaml` file to something else
(eg `name: darp-env`).  Then modify change the `CONDA_ENV` variable on line 3 in `setup.sh` to match 
(eg `CONDA_ENV="darp-env"`).

Then run the setup script.
```
./setup.sh
```

# Running
Before running anything make sure the correct Conda environment is selected:
```
conda activate darp
```
There are two script files: `data_indices.py` and `darp_restricted_fragments.py`.  

The main script is `darp_restricted_fragments.py` and requires one mandatory argument, which is an integer index into 
the problem set.  The available instances and their corresponding index can be viewed with the other script:
```
python data_indices.py -m all
0 a2-16
1 a2-20
2 a2-24
3 a3-18
4 a3-24
5 a3-30
...
```
So to solve instance `a2-16`, run
```
python -O darp_restricted_fragments.py 0
```
For the instances used by Gwschind and Irnich (2015), which extend the time windows of requests, use the `--extend`
parameter with a value of `3`.  For example,
```
python -O darp_restricted_fragments.py --extend 3 0
```
solves `a2-16` with widened time windows.  Note the `-O` flag to Python, which skips debug assertions.  Performance
will suffer if this flag is not supplied.

The `darp_restricted_fragments.py` script will create log directories under `./logs/darp/rf/`. 

Further documentation can be found using the `--help` flags on both scripts.

