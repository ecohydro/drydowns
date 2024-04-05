#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2024'

__license__ = 'MIT'
__date__ = 'Thu 04 Apr 24 13:17:39'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           soil.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2024


"""


# IMPORTS
import numpy as np


# VARIABLES

rho = 1000.
g = 9.8
PRECISION = 2

field_capacity = -33 / 1000 # Field capacity in MPa.

soils = {
    'sand':{
        'b': 4.05,
        'Psi_S_cm': 12.1,   # saturated water tension, cm
        'Psi_l_cm': 4.66,   # leakage water tension, cm
        'n': 0.395,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 1.056,        # saturated hydraulic conductivity, cm/min
        'S': 1.52           # sorptivity, cm/min^1/2    
    },
    'loamy sand':{
        'b': 4.38,
        'Psi_S_cm': 9.0,    # saturated water tension, cm
        'Psi_l_cm': 2.38,   # leakage water tension, cm
        'n': 0.410,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.938,        # saturated hydraulic conductivity, cm/min
        'S': 1.04           # sorptivity, cm/min^1/2  
    },
    'sandy loam':{
        'b': 4.90,
        'Psi_S_cm': 21.8,   # saturated water tension, cm
        'Psi_l_cm': 9.52,   # leakage water tension, cm
        'n': 0.435,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.208,        # saturated hydraulic conductivity, cm/min
        'S': 1.03           # sorptivity, cm/min^1/2  
    },
    'silt loam':{
        'b': 5.30,
        'Psi_S_cm': 78.6,   # saturated water tension, cm
        'Psi_l_cm': 75.3,   # leakage water tension, cm
        'n': 0.485,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0432,       # saturated hydraulic conductivity, cm/min
        'S': 1.26           # sorptivity, cm/min^1/2  
    },
    'loam':{
        'b': 5.39,
        'Psi_S_cm': 47.8,   # saturated water tension, cm
        'Psi_l_cm': 20.0,   # leakage water tension, cm
        'n': 0.451,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0417,       # saturated hydraulic conductivity, cm/min
        'S': 0.693          # sorptivity, cm/min^1/2  
    },
    'sandy clay loam':{
        'b': 7.12,
        'Psi_S_cm': 29.9,   # saturated water tension, cm
        'Psi_l_cm': 11.7,   # leakage water tension, cm
        'n': 0.420,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0378,       # saturated hydraulic conductivity, cm/min
        'S': 0.488          # sorptivity, cm/min^1/2  
    },
    'silty clay loam':{
        'b': 7.75,
        'Psi_S_cm': 35.6,   # saturated water tension, cm
        'Psi_l_cm': 19.7,   # leakage water tension, cm
        'n': 0.477,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0102,       # saturated hydraulic conductivity, cm/min
        'S': 0.310          # sorptivity, cm/min^1/2  
    },
    'clay loam':{
        'b': 8.52,
        'Psi_S_cm': 63.0,   # saturated water tension, cm
        'Psi_l_cm': 48.1,   # leakage water tension, cm
        'n': 0.476,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0147,       # saturated hydraulic conductivity, cm/min
        'S': 0.537          # sorptivity, cm/min^1/2  
    },
    'sandy clay':{
        'b': 10.4,
        'Psi_S_cm': 15.3,   # saturated water tension, cm
        'Psi_l_cm': 8.18,   # leakage water tension, cm
        'n': 0.426,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0130,       # saturated hydraulic conductivity, cm/min
        'S': 0.223          # sorptivity, cm/min^1/2  
    },
    'silty clay':{
        'b': 10.4,
        'Psi_S_cm': 49.0,   # saturated water tension, cm
        'Psi_l_cm': 23.0,   # leakage water tension, cm
        'n': 0.492,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0062,       # saturated hydraulic conductivity, cm/min
        'S': 0.242          # sorptivity, cm/min^1/2  
    },
    'clay':{
        'b': 11.4,
        'Psi_S_cm': 40.5,   # saturated water tension, cm
        'Psi_l_cm': 24.3,   # leakage water tension, cm
        'n': 0.482,         # porosity, cm^3/cm^3 (is Psi_S) in C&H,
        'Ks': 0.0077,       # saturated hydraulic conductivity, cm/min
        'S': 0.268          # sorptivity, cm/min^1/2  
    }
}

# SOIL CLASS

class Soil():
    """ Defines a soil object based on either passed parameters
    or a soil texture class corresponding to the textures defined in 
    Clapp & Hornberger (C&H), 1978, Table 2.

    Usage: Soil(texture, params)
        Notes: 
            If texture is not provided, params must be included.
            If texture is provided, params is ignored.
            Capitilization in texture classes is ignored.
        
        texture = texture name from Clapp & Hornberger, 1978, Table 2.
            Valid options are:
            ["Sand", "Loamy Sand", "Sandy Loam", "Silt Loam", "Loam",
             "Sandy Clay Loam", Silty Clay Loam", "Clay Loam", "Sandy Clay",
             "Silty Clay", "Clay"]
        
        params = dictionary containing values for the soil parameters:
            params = {
                'b': 11.4,
                'Psi_S': 40.5,  # saturated water tension, cm
                'Psi_l': 24.3,  # leakage water tension, cm
                'n': 0.482,     # porosity, cm^3/cm^3 (is Psi_S) in C&H,
                'Ks': 0.0077,   # saturated hydraulic conductivity, cm/min
                'S': 0.268      # sorptivity, cm/min^1/2  
            }
    
    Note: In C&H 1978, soil water retention relationships were defined according to _tensions_. 
    These tensions are specified as lengths, and are always _positive_ 
    (tension, like depth has an implied relationship to zero).

    To convert a tension, Psi_cm, (positive quantity of length) into a water potential, Psi_Pa, 
    (negative measure of energy density per unit volume, or Pa), you do the following:

    Psi_Pa = -1 * Psi_cm * rho * g

    This conversion is done during initiation of the soil class.

    """
    def __init__(self, texture=None, params=None):
        """ Initializes a soil object.

        The init function requires _either_ a soil texture or a params dictionary
        (see class description)
        
        """
        self._valid_params = set(['b', 'Psi_S_cm', 'Psi_l', 'n', 'Ks', 'S'])
        self._required_params = set(['b', 'Psi_S_cm', 'n', 'Ks'])
        
        # Set required attributes to None:
        [setattr(self, attr, None) for attr in self._required_params]
        
        if texture: # If this class is instanced with a specific USDA soil texture.
            texture = texture.lower() # Force the soil texture category to lower case
            # Assign texture parameters based on the appropriate soil class:
            for attr, val in soils[texture].items():
                setattr(self, attr, val)
        elif params: # If the class is instanced with a set of soil parameters
            for attr, val in params.items():
                # Only include valid soil parameters
                if attr in self._valid_params:  
                    setattr(self, attr, val)
            # Check that all required parameters have been set
            if not self._required_params.issubset(self.__dict__.keys()):
                missing = self._required_params.difference(self.__dict__.keys())
                raise AttributeError("Missing required parameters, {list}".format(list=missing))
        else: 
            raise AttributeError("Must pass either a soil texture or dict of parameters")
        
        # Set Psi_S (MPa) from Psi_S_cm (cm). Assumes that Psi_S_cm is positive (as it should be!)
        self.Psi_S_MPa = -1 * self.Psi_S_cm / 100 * rho * g / 1E6 
        
        # Set Ks (mm/day) from Ks (cm/min).
        self.Ks = self.Ks*10*60*24

        # This version of sfc calculation comes from Laio et al. 2001b. Specifically, cf. the discussion
        # on p.714, and equation 15. 
        # self.sfc = pow(0.05/(self.Ks/10),1/(2*self.b+3))  # Convert Ks in cm/day 
        
        # Calculate volumetric field capacity
        self.theta_fc = self.theta(psi=field_capacity)

        # This version of the sfc calculation uses the psi-theta relationships in Clapp & Hornberger to 
        # determine s_fc based on a texture-specific field_capacity.
        self.sfc = self.s(psi=field_capacity)
        
        # Make sure that field capacity is always lower than soil porosity.
        if self.sfc > 1:
            raise ValueError("soil field capacity, {sfc} is larger than 1".format(
                sfc=self.sfc
            ))

        # Set parameters related to pore size distribution index:
        self.Beta = 2*self.b + 4
        self.c = 2*self.b + 3

        # Hygroscopic point is when soil is so dry no further evaporation will occur.
        self.sh = self.s(self.theta(-10))               # Hygroscopic point in relative soil moisture [0-1]
        self.nZr = None                                 # TODO: Hygroscopic point is a wonky parameter stuck in the middle code.. consider setting elsewhere

    def _check_nZr(self):
        error = "Error: Calculation depends on value of self.nZr before calling self.set_nZr"
        if not self.nZr:
            raise AttributeError(error)

    def _check_theta(self, theta):
        error = "theta, {theta}, must be be in the interval (0,{n}]".format(
                theta=theta, n=self.n)
        if theta > self.n or theta < 0:
            raise ValueError(error)

    def psi(self, theta):
        """ Return water potential in Pa based 
        on volumetric soil water content in m^3/m^3

        Note: Assumes that Psi is a water potential, and therefore Psi < 0 for unsaturated soils!

        Usage: psi(theta):
        
            theta = soil water content [m^3/m^3]
        
        """
        self._check_theta(theta)
        s = self.s(theta=theta)          
        return round(self.Psi_S_MPa * pow(s,-self.b),PRECISION)
    
    def theta(self,psi):
        """ Return a volumetric water content in m^3/m^3 
        based on a given water potential (MPa)

        Note: 
        Usage: theta(psi):

            psi = soil water potential [MPa]
        
        """
        if psi > 0:
            raise ValueError("psi, {psi}, must be less than or equal to zero.".format(psi=psi))
        # Ensure result is rounded to correct precision and that we do not exceed porosity
        return min([round((self.n * pow(psi/self.Psi_S_MPa, 1/-self.b)),PRECISION), self.n]) 

    def s(self, theta=None, psi=None):
        """ Return a relative soil moisture value, s [0-1]
        given a volumetric water content [m^3/m^3] or a 
        water potential [MPa]

        Usage: s(theta):

            theta = volumetric water content [m^3/m^3]
            psi = water potential [MPa]

        Note: theta must be in the interval 0-n (porosity)
        Note: psi must be negative
        Note: Function must be called with either theta or psi, but not both.
        
        """
        if theta and psi:
            raise ValueError(
            "Both theta ({theta}) and psi {psi} values provided only one argument allowed".format(
                theta=theta,
                psi=psi
            ))
        if psi:
            theta = self.theta(psi)
        self._check_theta(theta)
        try:
            return round(theta/self.n, PRECISION)
        except:
            raise ValueError("Either theta or psi must be provided as an argument.")

    def set_nZr(self, z):
        """ Sets the nZr for this soil in order to 
        determine fluxes in mm/day rather than relative
        soil moisture

        Usage: set_nZr(plant)

            plant = plant object with plant.Zr value set.
        
        Returns:

            nZr = n * Zr

            Also sets internal soil property nZr according to:

                self.nZr = self.n * plant.Zr
        """
        self.nZr = self.n * z
        return self.nZr 


