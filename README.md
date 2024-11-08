# drydowns
A python package for identifying and characterizing drydown events from soil moisture data as detailed in:

> Morgan, B.E., Araki, R., Trugman, A.T., Caylor, K.K. (*in review*). Ecological and hydroclimatic determinants of vegetation water-use strategies. *Nature Ecology & Evolution*.


## Getting started
Example usage of this package is detailed in the [ismn-drydowns repository](https://github.com/ecohydro/drydowns). 

To edit the package:

1. Clone the repository
```bash
$ git clone git@github.com:ecohydro/drydowns.git
```

2. Create a virtual environment
```bash
$ cd drydowns
$ conda env create -f environment_linux.yml
$ conda activate ismn
```
Note that if using Windows, the `environment_win.yml` file should be used instead.

To install the package in editable format (for development alongside analysis):
```bash
pip install -e git+ssh://github.com/ecohydro/drydowns.git@v0.0.1
```

## Authors
[Ryoko Araki](https://github.com/RY4GIT), San Diego State University
[Bryn Morgan](https://github.com/brynemorgan), UC Santa Barbara

## Contact
Bryn Morgan, [brynmorgan@ucsb.edu](mailto:brynmorgan@ucsb.edu)