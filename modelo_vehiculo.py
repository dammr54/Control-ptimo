# Modelo dinámico de un vehiculo de traccion diferencial (2D / movimiento direccional en el plano "terrestre")
# Vector de estado
# x1: x (posicion en coordenadas del mundo) 
# x2: y (posicion en coordenadas del mundo) 
# x3: theta (orientación en coordenadas del mundo relativa al eje x) 
# x4: velocidad lineal que se puede descomponer en ambos ejes como v_i = v cos(x3) y v_j = v sen(x3)
# v_i (velocidad longitudinal del auto, referida al eje longitudinal del auto (pitch axis), + nariz)
# v_j (velocidad lateral del auto, referida al eje lateral del auto (pitch axis), + ala izquierda)
# x5: omega (velocidad de giro entorno al eje vertical del auto (yaw axis), + hacia la ziquierda)

# Vector de control: torque aplicado a los actuadores
# u1: Tr (torque derecho)
# u2: Tl (torque izquierdo)

# Vector de ruido en actuadores y perturbaciones en estado
# v1: n_xk
# v2: n_zeta
# v3: velocidad del viento
# v4: dirección del viento relativa al eje longitudinal del avión

import numpy as np
from params_vehiculo import *

def modelo_vehiculo(t, x, u, sigma_vx):
    # restricciones de control 

    # actualizacion de angulo en cada vuelta
    if x[2] > 2*np.pi:
        x[2] = x[2] - 2*np.pi
    elif x[2] < -2*np.pi:
        x[2] = x[2] + 2*np.pi
    
    # perturbacion v ???
    v = sigma_vx*np.random.randn(len(sigma_vx)) # '*' is the element-wise multiplication
                                              # when using NumPy arrays.
    
    cos_x2 = np.cos(x[2])
    sen_x2 = np.sin(x[2])

    largo_u = len(u)
    if largo_u == 5:
        ampliado = True
    else:
        ampliado = False
    if ampliado == True:
        xdot = np.array([
            x[3]*cos_x2,  # x_dot
            x[3]*sen_x2,  # y_dot
            x[4],           # theta_dot
            1/(m*r)*(u[0] + u[1]) - c/m*x[3],       # velocidad lineal
            W/(2*J*r)*(u[0] - u[1]) - b/J*x[4],      # omega_dot
            u[2] - x[0],
            u[3] - x[1],
            u[4] - x[2]])
    elif ampliado == False:
        xdot = np.array([
             x[3]*cos_x2,  # x_dot
             x[3]*sen_x2,  # y_dot
             x[4],           # theta_dot
             1/(m*r)*(u[0] + u[1]) - c/m*x[3],       # velocidad lineal
             W/(2*J*r)*(u[0] - u[1]) - b/J*x[4]]) + v     # omega_dot
    return xdot

def main(): # Test modelo_avion
    np.random.seed(0)

    t = 0.
    # x = (x=1, y=0, theta=45°, v=100, w=0)
    x = np.array([1., 0., np.pi/4., 100, 0.]) 
    # u = (tr=1, tl=0.1)
    u = np.array([1., 0.1])
    # sigma_v = 
    sigma_v = np.array([0.1,0.05,0.2,np.pi])
    # x_dot = (vx, vy, w, a, dot_w)
    x_dot = modelo_vehiculo(t, x, u, sigma_v)
    print(x_dot)
    
if __name__ == '__main__': main()