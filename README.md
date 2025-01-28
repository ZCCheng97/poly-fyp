# poly-fyp
## Introduction
Title: Ionic Conductivity Prediction of Solid Polymer Electrolytes Using Transformer Representations

An Honours Project 1 (FYP) submitted to the School of Chemistry, Chemical Engineering and Biotechnology, Nanyang Technological University in partial fulfillment of the requirements for the degree of Bachelor of Science in Chemistry and Biological Chemistry.

## Setup
### Installing all dependencies:

```
$ pip install -r requirements.txt
```

### Requirements:

```
Python version: 3.12.2
OS Platform: Windows
OS Version: Windows 10
```
All package requirements can be found in `requirements.txt`.

## Run instructions
Use `run.sh` to execute the files after installing required packages. You can pass the function name of each step of the pipeline consecutively as arguments when calling `main.py`. By default it currently runs all steps from `data_cleaning` to `cv`.

Make use of `src/config.py` to change run settings. 

## Data Source and Reference
1. Chemistry-Informed Machine Learning for Polymer Electrolyte Discovery
```bibtex
@article{doi:10.1021/acscentsci.2c01123,
author = {Bradford, Gabriel and Lopez, Jeffrey and Ruza, Jurgis and Stolberg, Michael A. and Osterude, Richard and Johnson, Jeremiah A. and Gomez-Bombarelli, Rafael and Shao-Horn, Yang},
title = {Chemistry-Informed Machine Learning for Polymer Electrolyte Discovery},
journal = {ACS Central Science},
volume = {9},
number = {2},
pages = {206-216},
year = {2023},
doi = {10.1021/acscentsci.2c01123},

URL = { 
    
        https://doi.org/10.1021/acscentsci.2c01123
    
    

},
eprint = { 
    
        https://doi.org/10.1021/acscentsci.2c01123
    
    

}

}

```
