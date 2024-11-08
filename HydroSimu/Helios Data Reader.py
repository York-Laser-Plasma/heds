#Helios data reader .exo files


import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy.integrate import trapezoid
from scipy.ndimage import uniform_filter1d

# Load data
file = r"C:\Users\benny\Downloads\IPA_1mm_500zn.exo"
data = Dataset(file, 'r')

# Read variables
Time = data.variables['time_whole'][:]
Radius = data.variables['zone_boundaries'][:]
Density = data.variables['mass_density'][:]
ElectronDensity = data.variables['elec_density'][:]
IonTemp = data.variables['ion_temperature'][:]
ElecTemp = data.variables['elec_temperature'][:]
Mass = data.variables['zone_mass'][:]
Pressure = data.variables['ion_pressure'][:] + data.variables['elec_pressure'][:]
Velocity = data.variables['fluid_velocity'][:] / 100000  # Converted to km/s
Volume = Mass / Density

X,Y = np.meshgrid(Time[:], Radius[0,:])

Z = Density[:-1,:]
plt.pcolormesh(X,Y,Z)
plt.show()

#The Variable arrays are stored in 3D [0,1,2] 0 = time , 1 = zone , 2 = variable quantity i.e Density
#plt.imshow((Density[:,0] *1e-9, Density[0,:] * 10000, Density[:,:]))
#plt.pcolormesh(Density[:,0] *1e-9, Density[0,:] * 10000, Density[:-1,:-1], shading='auto', cmap='jet')
#plt.colorbar(label='Mass Density (g/cm^3)')
#plt.show()

x = np.arange(0,10,1)
y = np.arange(0,10,1)

X, Y = np.meshgrid(x,y)
z = Y+X

plt.pcolormesh(X,Y,z)
plt.show()





'''
try:
    Flux = data.variables['Freq. resolved net flux at region interfaces [J per (cm2.sec.eV)]'][:]
    Groups = data.variables['photon_energy_group_bounds'][:]
    
    LaserEnergy = data.variables['LaserEnDeliveredTimeInt'][:]
    LaserPower = np.concatenate([[0], np.diff(LaserEnergy) / np.diff(Time)])
    SimulationPower = LaserPower

    DepositedLaserPowerZones = data.variables['LaserPwrSrc'][:]
    DepositedLaserPower = np.sum(DepositedLaserPowerZones * Mass, axis=0)
    DepositedEnergy1 = np.trapz(DepositedLaserPower, Time)  # Using numpy's trapezoid method
    DepositedEnergy = data.variables['LaserEnTimeIntg'][:] * Volume
    DepositedEnergy = np.sum(DepositedEnergy[:, -1])
    
    # Laser statistics
    AdjustedEnergy = LaserEnergy * np.pi * (0.06 / 2)**2
    LaserIntensity = LaserPower
    RealLaserPower = LaserIntensity * np.pi * (0.06 / 2)**2
    DepositedLaserPower = DepositedLaserPower * np.pi * (0.02)**2
    MaxLaserIntensity = np.max(LaserIntensity)
except KeyError:
    pass

IonTempKelvin = 1.16E4 * IonTemp
FoamDensity = Density[-1, 0]

# Shock front calculation
Shock = (Density - Density[:, 0][:, np.newaxis]) > 0.1
FoamMaxRadius = np.max(Radius[:, 0])
BulkRegions = Radius[:-1, 0] < FoamMaxRadius
Shock *= BulkRegions[:, np.newaxis]

ShockIndex = [np.argmax(Shock[:, t]) if np.any(Shock[:, t]) else 1 for t in range(Shock.shape[1])]
ShockRadius = Radius[ShockIndex, np.arange(len(Time))]

InterfaceIndex = np.argmax(np.abs(np.diff(Density[:, 0])) > 1)
InterfaceRadius = Radius[InterfaceIndex, :]

ShockVelocity = uniform_filter1d(np.diff(ShockRadius) / np.diff(Time), size=10)
InterfaceVelocity = np.diff(InterfaceRadius) / np.diff(Time)
ParticleVelocity = Velocity[InterfaceIndex, :]

plt.figure()
DensityPlot = plt.pcolormesh(Time * 1e9, Radius[:-1, :] * 10000, Density, shading='auto', cmap='jet')
plt.colorbar(label='Mass Density (g/cm^3)')
plt.title('Density Plot')
plt.xlabel('Time (ns)')
plt.ylabel('Radius (μm)')
plt.ylim([0, 1295])
plt.yscale('log')

plt.figure()
plt.plot(Time * 1e9, ShockRadius * 10000, label='Shock radius')
plt.plot(Time * 1e9, InterfaceRadius * 10000, label='Interface radius')
plt.xlabel('Time (ns)')
plt.ylabel('Radius (μm)')
plt.legend()

plt.figure()
plt.plot(Time[:-1] * 1e9, ShockVelocity / 1e5, label='Shock velocity')
plt.plot(Time[:-1] * 1e9, InterfaceVelocity / 1e5, label='Interface velocity')
plt.xlabel('Time (ns)')
plt.ylabel('Velocity (km/s)')
plt.legend()

plt.figure()
plt.plot(Time * 1e9, ParticleVelocity, label='Particle velocity')
plt.plot(Time[:-1] * 1e9, InterfaceVelocity / 1e5, label='Interface velocity')
plt.xlabel('Time (ns)')
plt.ylabel('Velocity (km/s)')
plt.legend()

plt.figure()
IonTempKelvinPlot = plt.pcolormesh(Time * 1e9, Radius[:-1, :] * 10000, IonTempKelvin, shading='auto', cmap='jet')
plt.colorbar(label='Ion Temperature (K)')
plt.title('Ion Temperature Plot')
plt.xlabel('Time (ns)')
plt.ylabel('Radius (cm)')
plt.ylim([0, 1295])
plt.yscale('log')
'''
plt.show()
