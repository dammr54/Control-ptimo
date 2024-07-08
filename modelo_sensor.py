# Modelo del sensor: implementa solamente un GPS
# Vector de estado
# Vector de estado
# x1: x (posicion en coordenadas del mundo) 
# x2: y (posicion en coordenadas del mundo) 
# x3: theta (orientación en coordenadas del mundo relativa al eje x) 
# x4: velocidad lineal que se puede descomponer en ambos ejes como v_i = v cos(x3) y v_j = v sen(x3)
# x6: omega (velocidad de giro entorno al eje vertical del auto (yaw axis), + hacia la ziquierda)
#
# Vector de control
# u1: Torque rueda derecha
# u2: torque rueda izquierda
#
# Vector de ruido en el sensor
# w1: n_x GPS
# w2: n_y GPS
# w3: n_theta_sensor GPS
# w4: n_v encoder
# w5: n_w encoder o imu
# w6: n_a imu

import numpy as np
from params_vehiculo import *
#
def modelo_sensor(x, sensor):
    w = sigma_wg*np.random.randn(len(sigma_wg))*0
    if sensor == 'ninguno': 
        z = np.array([x[0],
                    x[1],
                    x[2]])
    elif sensor == 'GPS':
        z = np.array([x[0],
                    x[1],
                    x[2]]) + w
    elif sensor == 'encoder':
        z = np.array([x[0],
                    x[1],
                    x[2]])
    elif sensor == 'IMU':
        z = np.array([x[0],
                    x[1],
                    x[2]])
    return z

def main(): # Test modelo_sensor
    np.random.seed(0)

    # x = (x=1, y=0, theta=45°, v=100, w=0)
    x = np.array([1., 0., np.pi/4., 100., 0.]) 
    sensor = 'GPS'
    z = modelo_sensor(x, sensor)
    print(z)
    
if __name__ == '__main__': main()