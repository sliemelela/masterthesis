# Master Thesis

## Start up
This codebase is entirely written in Python 3.7.6. Make sure to use virtual environments to ensure consistent results of the program.


### Setting up a virtual environment
There are many ways of setting up a virtual environment. The demenonstration below is just one of many ways.
To get started, navigate to your project folder and create

```bash
mkdir Environments
```
and navigate to this folder. If you have not done this yet, make sure to install `virtualenv` using

```bash
pip install virtualenv
```

You can now create a new virtual environment using the command

```bash
virtualenv your_favorite_name
```
where `your_favorite_name` is some name you give the virtual environment.

To activate your new virtual environment, use the command
```bash
source your_favorite_name/bin/activate
```

If you would like to use another virtual environment or stop using it, you simply use the command 
```bash
deactivate
```

## Installing the package need in the Inverse Optimization algorithm
To make sure you are able to properly run the notebooks, you need to make sure you install the (local) package `inverse_optim`.
Move to the folder containing setup.py and run 
```bash
pip install --editable .
```
