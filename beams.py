import numpy as np
import scipy
from scipy import special

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# This program doens´t have support from unit system, but every value for variables respects the SI unit as show above:
# W = 1 (1 m), W = 0.01 (1 cm), W = 0.001 -> (1 mm)

# Gaussian beam Width
def W( z, W0, lamb ):

    z0 = W0**2 * (np.pi/lamb)

    return  W0*( 1 + (z/z0)**2 )**0.5

# Gaussian beam´s wavefront radius of curvature
def R(z, W0, lamb):

    z0 = (W0**2) * (np.pi/lamb)

    return z*( 1 + (z0/z)**2)

# Gaussian´s beam Phase Angle
def Zeta( z, z0 ):

    return np.arctan( z/z0 )

# Gaussian´s beam Intensity 
def Gaussian_beam( z, W0, A0, lamb ):

    W1 = W(z, W0, lamb)

    a = (W0/W1)**2

    rho = np.linspace( -3*W1, 3*W1, 500 )

    b = np.exp( - 2*(rho**2/W1**2) )

    I0 = np.abs(A0)**2

    I = I0*a*b

    return I

# Hermite-Gauss Polynomial
def HGl(l, u):
    return special.hermite(l, monic=True)(u) * np.exp(-u**2/2)

# Hermite-Guass beam Intensity
def Hermite_Gaussian_beam(z, W0, Alm, lamb, l, m):
    
    W1 = W(z, W0, lamb)
  
    x = np.linspace(-3*W0, 3*W0, 100)
    y = np.linspace(-3*W0, 3*W0, 100)
    
    X, Y = np.meshgrid(x, y)

    a = (W0/W1)**2
    
    Ilm = np.abs(Alm)**2
    
    hgl_x = HGl(l, np.sqrt(2)*X/W1)
    
    hgl_y = HGl(m, np.sqrt(2)*Y/W1)

    return Ilm * a * (hgl_x**2) * (hgl_y**2), x, y

# Bessel beam Intensity
def Bessel_beam(m, Am, k, Rho):

    Im = np.abs( Am )**2

    kT = np.sqrt( k**2 - m**2 )

    Jm = special.jn( m, kT*Rho )

    return Im*Jm**2

# Laguerre´s generalised polynomial
def Lml( l, m, u ):
    return special.genlaguerre( m, l, monic=True )(u)

# Laguerre-Gauss beam Intensity
def Laguerre_Gaussian_beam(z, X, Y, Alm, W0, lamb, l, m):

    Rho = np.sqrt(X**2+Y**2)

    W1 = W(z, W0, lamb)

    a = (W0/W1)**2

    Ilm = np.abs( Alm )**2

    b = (Rho/W1)**l

    L = Lml( l, m, 2*(Rho**2)/(W1**2) )

    return Ilm * a * (b**2) * (L)**2 * np.exp( -2*(Rho**2)/(W1**2) )

# Gaussian Beam simulation
# --------------------------- Standard Values ---------------------------

# Wavelengh
lamb = 633e-9

# Beam Waist
W0 = 0.001/2

# Raylengh range
z0 = W0**2 * (np.pi/lamb)

# Wavelengh number
k = 2*np.pi/lamb

# Initial amplitude
A0 = 1

# Guassian Beam Divergence angle
theta0 = 2*lamb/(2*W0*np.pi)

# z axis range
z = np.linspace( -3*z0 , 3*z0, 500)

# Beam Radius range
rho = np.linspace(-3*W0, 3*W0, 500)

# --------------------------- Standard Values ---------------------------

I = Gaussian_beam(z, W0, A0, lamb)

fig, ax = plt.subplots( 1, 1, figsize=( 12, 6 ) )

ax.plot(z, W(z, W0, lamb), color='red')
ax.plot(z, -W(z, W0, lamb), color='red')

ax.set_xlim( -np.sqrt(2)*z0, np.sqrt(2)*z0 )
ax.set_ylim( -3*W0, 3*W0 )

ax.set_title( 'Gaussian Beam' )
ax.set_xlabel('z')
ax.set_ylabel('W(z)')
ax.axvline(x=z0, color='blue', linestyle=':', label='$z_0$')
ax.axvline(x=-z0, color='blue', linestyle=':', label='$-z_0$')
ax.axhline(y=np.sqrt(2)*W0, color='orange', linestyle=':', label='$W_0\sqrt{2 }$')
ax.axhline(y=-np.sqrt(2)*W0, color='orange', linestyle=':', label='$-W_0\sqrt{2 }$')
ax.plot( z, theta0*z, color='red', linestyle=':', label='Beam Div' )
ax.plot( z, -theta0*z, color='red', linestyle=':', label='Beam Div' )

bcm = ax.pcolormesh( z, rho, I, cmap='plasma', shading='auto' )

ax.legend()
ax.grid()
cbar = plt.colorbar(bcm, label='Intensity')
plt.show()

fig = plt.figure(figsize=(10, 6))

Z, Rho = np.meshgrid(z, rho)

ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Z, Rho, I, cmap='plasma', rstride=5, cstride=5, alpha=0.8, linewidth=0, antialiased=False)
cbar = fig.colorbar(surf, shrink=0.5, aspect=10, label='Intensity')
ax.set_xlabel('z')
ax.set_ylabel('rho')
ax.set_zlabel('Intensity')
ax.set_title('3D Surface (Gaussian Beam)')
plt.show()

# Hermite-Gauss Beam simulation
# --------------------------- Standard Values ---------------------------

# Hermite-Gauss modes of vibration
l, m = 12, 15

# Initial amplitude
Alm = 1

# Wavelengh
lamb = 633e-9

W0 = 0.001/2

k = 2*np.pi/lamb

z_values = np.linspace(-2*z0, 2*z0, 100)
# --------------------------- Standard Values ---------------------------


fig_hermite, ax_hermite = plt.subplots(1, 1, figsize=(10, 6))

I, x, y = Hermite_Gaussian_beam(z_values, W0, Alm, lamb, l, m)

contour_hermite = ax_hermite.contourf(x, y, I, cmap='plasma', levels=100)
ax_hermite.set_title(f'Hermite-Gaussian Beam Intensity Distribution mode ({l},{m})')
ax_hermite.set_xlabel('X')
ax_hermite.set_ylabel('Y')
cbar_hermite = plt.colorbar(contour_hermite, label='Intensity')
plt.show()

Y, X = np.meshgrid(x, y)

fig_hermite = plt.figure(figsize=(10, 6))

ax_hermite = fig_hermite.add_subplot(111, projection='3d')

surf = ax_hermite.plot_surface(X, Y, I, cmap='plasma', rstride=5, cstride=5, alpha=0.8, linewidth=0, antialiased=False)
cbar = fig_hermite.colorbar(surf, shrink=0.5, aspect=10, label='Intensity')

ax_hermite.set_xlabel('X')
ax_hermite.set_ylabel('Y')
ax_hermite.set_zlabel('Intensity')
ax_hermite.set_title(f'3D Surface (Hermite-Gaussian Beam) mode ({l},{m})')
plt.show()


# Laguerre-Gauss Beam simulation
# --------------------------- Standard Values ---------------------------

# Laguerre-Gauss vibration modes
l_laguerre, m_laguerre = 10, 5

# Initial amplitude
Alm = 1

lamb = 633e-9

W0 = 0.001/2

k = 2*np.pi/lamb
# --------------------------- Standard Values ---------------------------

fig_laguerre, ax_laguerre = plt.subplots(1, 1, figsize=(10, 6))

z_values = np.linspace(-z0, z0, 600)

W1 = W(z_values, W0, lamb)

rho = np.linspace( 0, W0, 600 )
phi = np.linspace( 0, 2*np.pi, 600 )

Rho, Phi = np.meshgrid(rho, phi)

x = np.linspace(-0.003, 0.003, 600)
y = np.linspace(-0.003, 0.003, 600)
X, Y = np.meshgrid(x, y)

I = Laguerre_Gaussian_beam(z_values, X, Y, Alm, W0, lamb, l_laguerre, m_laguerre)

contour_laguerre = ax_laguerre.contourf(rho, phi, I, cmap='plasma', levels=50)
ax_laguerre.set_title(f'Laguerre-Gaussian Beam Intensity Distribution mode ({l_laguerre},{m_laguerre})')
ax_laguerre.set_xlabel('X')
ax_laguerre.set_ylabel('Y')
cbar_laguerre = plt.colorbar(contour_laguerre, label='Intensity')
plt.show()

Rho, Phi = np.meshgrid( rho, phi )

fig_laguerre = plt.figure(figsize=(10, 6))

ax_laguerre = fig_laguerre.add_subplot(111, projection='3d')

surf = ax_laguerre.plot_surface(Rho, Phi, I, cmap='plasma', rstride=5, cstride=5, alpha=0.8, linewidth=0, antialiased=False)
cbar = fig_laguerre.colorbar(surf, shrink=0.5, aspect=10, label='Intensity')

ax_laguerre.set_xlabel('X')
ax_laguerre.set_ylabel('Y')
ax_laguerre.set_zlabel('Intensity')
ax_laguerre.set_title(f'3D Surface (Laguerre-Gaussian Beam) mode ({l_laguerre},{m_laguerre})')
plt.show()



# Bessel Beam simulation
# --------------------------- Standard Values ---------------------------

# Initial amplitude
Am = 1

# Bessel Special Function order
m_bessel = 0

lamb = 633e-9

k = 2*np.pi/lamb

x = np.linspace(-0.005, 0.005, 600)

y = np.linspace(-0.005, 0.005, 600)

X, Y = np.meshgrid(x, y)

theta = np.linspace(0, 2 * np.pi, 600)
# --------------------------- Standard Values ---------------------------

Rho=np.sqrt(X**2+Y**2)

I = Bessel_beam( m_bessel, Am, k, Rho)

fig_bessel = plt.figure(figsize=(10, 6))
ax_bessel = fig_bessel.add_subplot(111, projection='3d')
ax.set_facecolor('black')
Rho, Theta = np.meshgrid(rho, theta)

surf_bessel = ax_bessel.plot_surface(Rho, Theta, I, cmap='plasma', alpha=0.8, linewidth=0, antialiased=False)
cbar_bessel = fig_bessel.colorbar(surf_bessel, shrink=0.5, aspect=10, label='Intensity')

ax_bessel.set_xlabel('X')
ax_bessel.set_ylabel('Y')
ax_bessel.set_zlabel('Intensity')
ax_bessel.set_title(f'3D Surface (Bessel Beam Mode {m_bessel})')
plt.show()