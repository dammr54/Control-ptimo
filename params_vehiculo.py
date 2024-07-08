import numpy as np
# Parametros robot de traccion diferencial
Ts = 0.01 # tiempo de muestreo
L = 0.6 # largo base
W = 0.4 # ancho base
H = 0.15 # alto base
r = 0.15 # radio ruedas
m = 10 # masa del cuerpo de la base
# momento de inercia de la base alrededor del eje normal al terreno
J = 1 # m/12*(L^2 + W^2) + 2 J_w(W/2)^2 
c = 0.3 # efectos de friccion dinamica (roce aerodinamico y resistencia viscosa) sentido longitudinal
b = 0.3 # efectos de friccion dinamica (roce aerodinamico y resistencia viscosa) sentido eje de giro

# matrices 
# ruido en los sensores (tacometro) iid -> medicion de velocidades
sigma_wt = np.array([10**(-4), 10**(-4)])
Rt = np.array([[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 10**(-4), 0],
            [0, 0, 0, 0, 10**(-4)]])
# ruido en los sensores (encoder) 
sigma_we = (np.pi*r**2/360)*np.array([[1, 2/W], [2/W, 4/(W**2)]])
Re = (np.pi*r**2/360)*np.array([[0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 1, 2/W],
                                [0, 0, 0, 2/W, 4/W**2]])
# ruido en los sensores (IMU) 
sigma_wi = np.array([0.0004, 0.0004*np.pi**2])
Ri = np.array([[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0.0004, 0],
            [0, 0, 0, 0, 0.0004*np.pi**2]])

# ruido en los sensores (GPS) 
sigma_wg = np.array([0.01, 0.01, 2./360.**2*np.pi])
Rg = (np.pi*r**2/360)*np.array([[0.01, 0, 0, 0, 0],
                                [0,  0.01, 0, 0, 0],
                                [0, 0, 0, 2./360.**2*np.pi, 0],
                                [0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 0]])