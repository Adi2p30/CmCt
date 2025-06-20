# Calving - Aditya 

## Installation
To download 

you will need to do pip install pangeo-xesmf 
to install xesmf

xesmf has been used for its regridder and to use conservative regridding.


Format of data:

## ice_mask: ice_mask, annual mean percentage ice cover,
python


## time: time, time in years 
list of years from 2000 to 2020 (python list)

## x: cartesian x coordinate in meters
python list of integers

## y: cartesian y coordinate in meters
python list of integers




Satellite Data has 5x more resolution:
method - calculation shape: model(337, 577) sat(1680, 2880)

Tasks
1. Turn format of Satellite Data from %age to fraction or basically divide by 100
2. 