#!/usr/bin/env python

"""
Analyze the input runs for COM epicycle
behaviour. Outputs include:
COM drift in x-y

COM magnitude over time

COM with scri correction in x-y and magnitude,
plotted over the previous plots

COM with scri corrections vs rotating coordinate
frame unit vectors

COM with scri corrections dotted into the mentioned
unit vectors

Final corrected COM motion

Note that this is an analysis. No changes to 
the input data will be made by this script. the assumed
form for the COM is: COM = dx + vt + dr(cos(phi + dphi),
sin(phi+dphi), 0), where dx+vt is the scri correction.
"""

import sys
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import scipy.integrate as integrate
import h5py
import scri
import argparse

def DataLoader(h5location):
    #Given the h5 file location for the data, open the
    #h5 file and load the positions, masses, times of
    #the 2 black holes
    
    h5file = h5py.File(h5location,'r')
    xa = h5file['/AhA.dir/CoordCenterInertial.dat']
    xb = h5file['/AhB.dir/CoordCenterInertial.dat']
    mass_a = h5file['AhA.dir/ChristodoulouMass.dat'][:,1]
    mass_b = h5file['AhB.dir/ChristodoulouMass.dat'][:,1]
    
    return xa,xb,mass_a,mass_b,xa[:,0]

def NewtonianC(xa,xb,ma,mb):
    #Defines the Newtonian COM, c = c1+c2
    Nc_x = []
    Nc_y = []
    Nc_z = []
    for idx in range(0,len(xa)):
        Nc_x = np.append(Nc_x,(ma[idx]*xa[idx,1] + mb[idx]*xb[idx,1])/(ma[idx]+mb[idx]))
        Nc_y = np.append(Nc_y,(ma[idx]*xa[idx,2] + mb[idx]*xb[idx,2])/(ma[idx]+mb[idx]))
        Nc_z = np.append(Nc_z,(ma[idx]*xa[idx,3] + mb[idx]*xb[idx,3])/(ma[idx]+mb[idx]))
        
    return np.column_stack((Nc_x,Nc_y,Nc_z))

def MagC(x):
    #Computes the magnitude of the input vector, assuming Nx3 shape
    C = []
    for idx in range(0,len(x)):
        C = np.append(C,math.sqrt(x[idx,0]*x[idx,0] + x[idx,1]*x[idx,1] + x[idx,2]*x[idx,2]))
    return C

def COMcorrection(h5location,times):
    #Streamline COM corrections
    #Call scri function and make correction array to be subtracted from original data
    #The total Simulation Annex COM correction used beginning fraction = relaxed time/total time and 
    #ending fraction = 10%
    #Note that "Horizons.h5" is 11 characters
    
    #find relaxed time
    with open(h5location[:-11]+"metadata.txt","r") as metafile:
        for line in metafile:
            if "relaxed-measurement-time =" in line:
                tryfloat=[]
                for s in line.split():
                    try:
                        tryfloat.append(float(s))
                    except ValueError:
                        pass
                t_relax = tryfloat[0]
    print(h5location+'\n')
    bf = t_relax/times[-1]    #beginning fraction
    x,v,t1,t2 = scri.SpEC.estimate_avg_com_motion(h5location,bf,0.1) #corrective translation x and boost v
    
    return np.array([x + v*t_j for t_j in times])

def Compute_OrbitalFrequency(xA, xB):
    #Taken from SpEC/Support/bin/BbhDiagnosticsImpl.py
    #Changed default fit to Spline, from Fit
    #Took out unnecessary functions to cut down line space

    def SplineData(data):
        from scipy.interpolate import splrep,splev
        t   = data[:,0]
        spline_x = splrep(t, data[:,1])
        spline_y = splrep(t, data[:,2])
        spline_z = splrep(t, data[:,3])
        dx = splev(t, spline_x, der=1)
        dy = splev(t, spline_y, der=1)
        dz = splev(t, spline_z, der=1)
        v = np.vstack((t,dx,dy,dz)).T
        return data, v
    
    xA_fit,vA=SplineData(xA)
    xB_fit,vB=SplineData(xB)

    # Compute Orbital frequency (r x dr/dt)/r^2
    t  = xA_fit[:,0]
    dr = xA_fit[:,1:] - xB_fit[:,1:]
    dr2 = MagC(dr)**2   #changed from original SpEC function
    dv = vA[:,1:] - vB[:,1:]
    Omega = np.cross(dr,dv)/dr2[:,np.newaxis]
    return Omega

def UnitVectors(xa,xb):
    #Define the rotating unit vectors of the system
    #Given the 4-vector positions of the BBH
    sep = xa[:,[1,2,3]] - xb[:,[1,2,3]]

    n_hat = sep/MagC(sep)[:,np.newaxis]

    Ofreq = Compute_OrbitalFrequency(xa, xb)
    k_hat = Ofreq/MagC(Ofreq)[:,np.newaxis]

    lambda_hat = np.cross(-n_hat, k_hat)
    return n_hat, k_hat, lambda_hat

def Dot(veca, vecb):
    #take the dot product of each time index of the vectors veca, vecb
    #and store in a returned 1D vector. Assume 3D vector input
    
    d = []
    
    for idx in range(0,len(veca)):
        d = np.append(d, np.dot(veca[idx],vecb[idx]))
        
    return d

def EpicycleCorrection(n,l,ddotn,ddotl):
    #Apply the epicycle correction factor to the COM data,
    #given n,l,ddotn,ddotl
    
    return (np.multiply(n.T, ddotn) + np.multiply(l.T, ddotl)).T

def main():

    #Parse arguments from command line
    parser=argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dir",nargs="+",required="True",
                        help="Locations of Horizons.h5 files to consider.")
    parser.add_argument("--outfile",required="True",
                        help="Name for the output plots. The date is recommended.")

    args = parser.parse_args()

    for directory in args.dir:
        #Find all the Horizons.h5 files in the specified directory.
        in_files=[os.path.join(d,x)
                  for d, dirs, files in os.walk(directory, topdown=True)
                  for x in files if x.endswith('Horizons.h5')]

        plt.figure(1)
        plt.subplots(3,2,figsize=(15,20))
        ax1 = plt.subplot(321)  #COM position in x-y, org and scri corrected (delta)
        ax2 = plt.subplot(322)  #COM magnitude vs time, org and delta
        ax3 = plt.subplot(323)  # delta_x vs n_hat x, where n_hat is the unit vector of r in epicycles
        ax4 = plt.subplot(324)  # delta . k_hat cs delta . n_hat
        ax5 = plt.subplot(325)  # delta . n_hat vs delta . lambda_hat
        ax6 = plt.subplot(326)  #COM position in x-y, scri+epicycle correction

        for in_file in sorted(in_files):
            #For each Horizons.h5 file that we have in the specified directory:
            x_a,x_b,mass_a,mass_b,times = DataLoader(in_file)
            
            Nc = NewtonianC(x_a,x_b,mass_a,mass_b)
            NC = MagC(Nc)

            delta = Nc - COMcorrection(in_file,times)

            n_hat,k_hat,lambda_hat = UnitVectors(x_a,x_b)
            ddotn = Dot(delta, n_hat)
            ddotl = Dot(delta,lambda_hat)
            ddotk = Dot(delta,k_hat)
            
            dphi = np.array([math.atan2(ddotl[idx],ddotn[idx]) for idx in range(0,len(ddotn))])
            omega = MagC(Compute_OrbitalFrequency(x_a,x_b))

            finalc = delta - EpicycleCorrection(n_hat,lambda_hat,ddotn,ddotl)

            #Plot the data!
            if 'Lev1' in in_file:
                colour='r'
            elif 'Lev2' in in_file:
                colour = 'b'
            elif 'Lev3' in in_file:
                colour= 'g'
            elif 'Lev4' in in_file:
                colour='m'
            elif 'Lev5' in in_file:
                colour='c'
            else:
                colour = 'y'

            ax1.plot(Nc[:,0], Nc[:,1], label=in_file[-17:-12]+' org', color=colour, alpha = 0.25)
            ax1.plot(delta[:,0],delta[:,1], label=in_file[-17:-12]+' with scri',color=colour)

            ax2.plot(times, NC, label = in_file[-17:-12]+' org', color = colour, alpha = 0.25)
            ax2.plot(times, MagC(delta), label = in_file[-17:-12]+' with scri',color = colour)

            ax3.plot(delta[:,0], n_hat[:,0], label = in_file[-17:-12],color = colour,alpha = 0.5)
            
            ax4.plot(ddotn, ddotk, label = in_file[-17:-12],color = colour)
            ax5.plot(ddotl,ddotn,label=in_file[-17:-12], color=colour)
            ax6.plot(finalc[:,0], finalc[:,1], label=in_file[-17:-12],color=colour,alpha=.5)

        ax1.legend()
        ax1.set_title('COM position, x-y')
        ax2.legend()
        ax2.set_title('COM magnitude vs time')
        ax3.legend()
        ax3.set_title('deltax vs n_hatx')
        ax4.legend()
        ax4.set_title('delta dot k_hat vs delta dot n')
        ax5.legend()
        ax5.set_title('delta dot n_hat vs delta dot lambda_hat')
        ax6.legend()
        ax6.set_title('COM position, with scri and epicycle correction')
        plt.savefig(args.outfile+'_'+directory[-13:]+'.pdf')
        plt.clf()
        plt.close()

if __name__=="__main__":
    main()
