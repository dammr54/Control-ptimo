# Jacobianos de un auto para el movimiento lateral-direccional en el plano 2D ("plano terrestre")

# Modelo dinámico de un vehiculo de traccion diferencial (2D / movimiento direccional en el plano "terrestre")
# Vector de estado
# x1: x (posicion en coordenadas del mundo) 
# x2: y (posicion en coordenadas del mundo) 
# x3: theta (orientación en coordenadas del mundo relativa al eje x) 
# x4: velocidad lineal que se puede descomponer en ambos ejes como v_i = v cos(x3) y v_j = v sen(x3)
# x6: omega (velocidad de giro entorno al eje vertical del auto (yaw axis), + hacia la ziquierda)

# Vector de control: torque aplicado a los actuadores
# u1: Tr (torque derecho)
# u2: Tl (torque izquierdo)

# Vector de ruido en actuadores y perturbaciones en estado
# v1: n_x
# v2: n_y
# v3: n_theta
# v4: n_v
# v5: n_w
#
# Vector de ruido en el sensor
# w1: n_x GPS
# w2: n_y GPS
# w3: n_theta_sensor GPS
# w4: n_v encoder
# w5: n_w encoder o imu
# w6: n_a imu
#
# perturbaciones en la entrada
# u_r
# u_l


import numpy as np
from params_vehiculo import *

# A, B, G, C, D, H = jacobianos_vehiculo(x,u)
def jacobianos_vehiculo(x, u, jacob):
    cos_x2 = np.cos(x[2])
    seno_x2 = np.sin(x[2])
    
    A = np.array(# x0=x  x1=y  x2=theta  x3=v  x4=w
        [[0, 0, -x[3]*seno_x2, cos_x2, 0],
         [0, 0, x[3]*cos_x2, seno_x2, 0],
         [0, 0, 0, 0, 1],
         [0, 0, 0, -c/m, 0],
         [0, 0, 0, 0, -b/J]])
    A_lqi_a = np.array(# x0=x  x1=y  x2=theta  x3=v  x4=w
        [[0, 0, -x[3]*seno_x2, cos_x2, 0, 0, 0, 0],
         [0, 0, x[3]*cos_x2, seno_x2, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, -c/m, 0, 0, 0, 0],
         [0, 0, 0, 0, -b/J, 0, 0, 0],
         [-1, 0, 0, 0, 0, 0, 0, 0],
         [0, -1, 0, 0, 0, 0, 0, 0],
         [0, 0, -1, 0, 0, 0, 0, 0]])

    B = np.array(# u1=tr u2=tl
        [[0, 0],
         [0, 0],
         [0, 0],
         [1/(m*r), 1/(m*r)],
         [W/(2*J*r), -W/(2*J*r)]])
    B_lqi_a = np.array(# u1=tr u2=tl
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1/(m*r), 1/(m*r), 0, 0, 0],
         [W/(2*J*r), -W/(2*J*r), 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])
     
    G = np.array( # v1 v2 v3 v4 v5
        [[ 1, 0, 0, 0, 0],
         [ 0, 1, 0, 0, 0],
         [ 0, 0, 1, 0, 0],
         [ 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 1]])
    G_lqi_a = np.array( # v1 v2 v3 v4 v5
        [[ 1, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 1, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 1, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 1]])


    C1 = np.array(# x0=x  x1=y  x2=theta  x3=v  x4=w
        [[1, 0, 0, 0, 0],
         [0, 1, 0, 0, 0],
         [0, 0, 1, 0, 0]])
         
    C2 = np.array(# x0=x  x1=y  x2=theta  x3=v  x4=w 
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]])
    
    C_lqi_a = np.array(# x0=x  x1=y  x2=theta  x3=v  x4=w
        [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 0]])

    D = np.array(# u1=tr u2=tl
        [[0, 0],
         [0, 0],
         [0, 0]])
    D_lqi_a = np.array(# u1=tr u2=tl
        [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0]])
    
    H1 = np.array([#w1 w2 w3 w4 w5
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0]])
    
    H2 = np.array([#w1 w2 w3 w4 w5
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0],
         [0, 0, 0, 0, 1]])

    if jacob == 'GPS':
        return A, B, G, C1, D, H1
    elif jacob == 'IMU':
        return A, B, G, C2, D, H2
    elif jacob == 'LQI aumentado':
        return A_lqi_a, B_lqi_a, G_lqi_a, C_lqi_a, D_lqi_a, H1
    return A, B, G, C1, D, H

def main(): # Test jacobianos_vehiculo

    # x = (x=1, y=0, theta=45°, v=100, w=0)
    x = np.array([1., 0., np.pi/4., 100, 0., 0, 0]) 
    # u = (tr=1, tl=0.1)
    u = np.array([1., 0.1])
    
    A, B, G, C, D, H = jacobianos_vehiculo(x, u, 'LQI aumentado')
    print(A, B, G, C, D, H)
    
if __name__ == '__main__': main()
