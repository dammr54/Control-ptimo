import math
import numpy as np
from scipy.linalg import expm, solve_discrete_are
from jacobianos_vehiculo import *
from jacobianos_vehiculo import *
from step_model import *
from modelo_vehiculo import *
from modelo_sensor import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
# control discreto PID de angulo y distancia
class PIDControl:
    def __init__(self, Kpa, Kia, Kda, Kpd, Kid, Kdd, dt=0.01):
        self.dt = dt # periodo de muestreo
        # constantes controlador de angulo discreto
        self.Kpa_d = Kpa
        self.Kia_d = Kia*dt
        self.Kda_d = Kda/dt
        # constantes controlador de distancia euclidiana discreto
        self.Kpd_d = Kpd
        self.Kid_d = Kid*dt
        self.Kdd_d = Kdd/dt
        # errores previos
        self.prev_err1_a = 0
        self.prev_err2_a = 0
        self.prev_err1_d = 0
        self.prev_err2_d = 0
        # entradas previas
        self.prev_u_a = 0
        self.prev_u_d = 0

    def calcular_control(self, estado, ref):
        # calculo error posicion, calculo de referencia de angulo y error de 
        # angulo con actualziacion de valores camino mas corto
        refx = ref[0]
        refy = ref[1]
        err_posx = refx - estado[0]
        err_posy = refy - estado[1]
        err_d = np.sqrt(err_posx**2 + err_posy**2)
        ref_angle = math.atan2(err_posy, err_posx)
        err_a = ref_angle - estado[2]
        # logica de calculo distancia de giro mas corta
        if err_a < -np.pi:
            err_a = err_a + 2*np.pi
            ref_angle = ref_angle + 2*np.pi
        elif err_a > np.pi:
            err_a = err_a - 2*np.pi
            ref_angle = ref_angle - 2*np.pi
        ############################# controlador discreto ############################
        ua = self.prev_u_a + (self.Kpa_d + self.Kia_d + self.Kda_d)*err_a - (self.Kpa_d + 2*self.Kda_d)*self.prev_err1_a + self.Kda_d*self.prev_err2_a
        ud = self.prev_u_d + (self.Kpd_d + self.Kid_d + self.Kdd_d)*err_d - (self.Kpd_d + 2*self.Kdd_d)*self.prev_err2_d + self.Kdd_d*self.prev_err2_d
        ############################# controlador discreto ############################
        # ------------ control en cascada angulo despues distancia -----------
        #if np.abs(err_a) > 1/180*np.pi:
        #    u1 = ua
        #    u2 = - ua
        #else:
        #    u1 = ud
        #    u2 = ud
        # ------------ control en cascada angulo despues distancia -----------
        # ley de control
        u1 = ud + ua
        u2 = ud - ua
        # actualizacion de parametros
        self.prev_err2_a = self.prev_err1_a
        self.prev_err1_a = err_a
        self.prev_err2_d = self.prev_err1_d
        self.prev_err1_d = err_d
        self.prev_u_a = ua
        self.prev_u_d = ud
        senal_control = [u1, u2] # vector de control (torques derecho e izquierdo)
        return senal_control, ref_angle, err_posx, err_posy, err_a
    

def referencias(t, t1=1.05, t2=5, t3=10, t4=15):
    if t < t1:
        ref = [1, 1]
        #ref = [0, 2]
    elif t >= t1 and t < t2:
        ref = [-1, 1]
    elif t >= t2 and t < t3:
        ref = [-1, -1]
        #ref = [2, 0]
    elif t >= t3 and t < t4:
        ref = [1, -1]
    elif t >= t4:
        ref = [1, 1]
    return ref

def referencias_circular(t, k, frecuencia=1/10):
    x = 1 + np.cos(2 * np.pi * frecuencia * t[k])  # Coordenadas x
    y = 1 + np.sin(2 * np.pi * frecuencia * t[k])  # Coordenadas y
    return [x, y]

# controlador LQI
class LQIcontrol:
    def __init__(self, x0, P0, u0, sigma_v, sigma_w, Kpa, Kia, Kda, Kpd, Kid, Kdd, dt=0.01):
        self.dt = dt # periodo de muestreo
        # estado
        self.x = x0 # estado real
        self.x_k_k = x0 # estado estimado
        self.x_k1_k = x0 # estado predicho 
        # medicion
        #z_k = modelo_sensor(x0, sensor) # vector de medicion
        # matriz de covarianza del estado
        self.P_k_k = P0 # matriz de covarianza del estado estimado
        self.P_k1_k = P0 # matriz de covarianza del estado predicho
        # matrices de covarianza perturbaciones del estado y del sensor
        self.Q_k = np.diag(sigma_v**2)
        self.R_k = np.diag(sigma_w**2)
        # entrada
        self.u_prev = u0
        # matrices de costo
        self.Q_lqr = np.array([[0.005,0.,0.,0.,0., 0, 0, 0],
                               [0.,0.005,0,0.,0., 0, 0, 0],
                               [0.,0.,0.001,0,0.,0, 0, 0],
                               [0.,0.,0.,0.001,0.,0, 0, 0],
                               [0.,0.,0.,0.,0.008, 0, 0, 0],
                               [0.,0.,0.,0.,0, 1/1000000000**2, 0, 0],
                               [0.,0.,0.,0.,0, 0, 1/1000000000**2, 0],
                               [0.,0.,0.,0.,0, 0, 0, 1/(200000*np.pi)**2]])
        self.R_lqr = np.array([[0.5, 0.0, 0, 0, 0],
                               [0.0, 0.5, 0, 0, 0],
                               [0.0, 0, 10**15, 0, 0],
                               [0.0, 0, 0, 10**15, 0],
                               [0.0, 0, 0, 0, 10**15]])
        # constantes controlador de angulo discreto
        self.Kpa_d = Kpa
        self.Kia_d = Kia*dt
        self.Kda_d = Kda/dt
        # constantes controlador de distancia euclidiana discreto
        self.Kpd_d = Kpd
        self.Kid_d = Kid*dt
        self.Kdd_d = Kdd/dt
        # errores previos
        self.prev_err1_a = 0
        self.prev_err2_a = 0
        self.prev_err1_d = 0
        self.prev_err2_d = 0
        # entradas previas
        self.prev_u_pid = 0

        # integracion error
        self.int_err_posx = 0
        self.int_err_posy = 0
        self.int_err_a = 0

    def calcular_control(self, estado, ref, jacob):
        refx = ref[0]
        refy = ref[1]
        # computo de jacobianos modelo discreto
        A, B, G, C, D, H = jacobianos_vehiculo(estado, self.u_prev, jacob)
        Ad = expm(A*Ts)
        Bd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(B)
        Gd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(G)
        A = Ad
        B = Bd
        G = Gd
        # error 
        err_posx = refx - estado[0]
        err_posy = refy - estado[1]
        err_d = np.sqrt(err_posx**2 + err_posy**2)
        ref_angle = math.atan2(err_posy, err_posx)
        err_a = ref_angle - estado[2]
        # logica de calculo distancia de giro mas corta
        if err_a < -np.pi:
            err_a = err_a + 2*np.pi
            ref_angle = ref_angle + 2*np.pi
        elif err_a > np.pi:
            err_a = err_a - 2*np.pi
            ref_angle = ref_angle - 2*np.pi
        ref = np.array([refx, refy, ref_angle, 0, 0])
        self.int_err_posx += err_posx
        self.int_err_posy += err_posy
        self.int_err_a += err_a
        estado = [estado[0], estado[1], estado[2], estado[3], estado[4], self.int_err_posx, self.int_err_posy, self.int_err_a]

        u_pid_pos = (self.Kpd_d + self.Kid_d + self.Kdd_d)*err_d - (self.Kpd_d + 2*self.Kdd_d)*self.prev_err1_d + self.Kdd_d*self.prev_err2_d
        u_pid_a = (self.Kpa_d + self.Kia_d + self.Kda_d)*err_a - (self.Kpa_d + 2*self.Kda_d)*self.prev_err1_a + self.Kda_d*self.prev_err2_a
        self.prev_u_pid = self.prev_u_pid + np.array([u_pid_pos + u_pid_a, u_pid_pos - u_pid_a])
        ruido = np.random.normal(0, 0.0001, A.shape)  # media 0 y desviación estándar 0.1
        A = A + ruido
        P = solve_discrete_are(A, B, self.Q_lqr, self.R_lqr)
        # K = (R+B^T*P*B)^{-1}*(B^T*P*A+N^T), N=0 -> 
        K = np.linalg.inv(self.R_lqr+B.T.dot(P).dot(B)).dot(B.T).dot(P).dot(A)
        u_lqr = -K.dot(estado)
        self.u_prev = np.array([u_lqr[0] + self.prev_u_pid[0], u_lqr[1] + self.prev_u_pid[1], refx, refy, ref_angle])
        # restricciones


        # --- Actualiza el gráfico de simulación ---
        #x_r = np.array([[ref[0], ref[1], ref_angle]])
        #UpdatePlot(fig1, x, x_k_k, z_k, x_r)
        #lista_ref_x.append(refx)
        #lista_ref_y.append(refy)
        #lista_ref_a.append(ref_angle)
        #lista_error_posicion.append(err_d)
        #lista_error_rumbo.append(err_a)
        self.prev_err2_a = self.prev_err1_a
        self.prev_err1_a = err_a
        self.prev_err2_d = self.prev_err1_d
        self.prev_err1_d = err_d
        #fo = funcion_objetivo(x_k_k[k,:], u[k,:], Q_lqr, R_lqr, ref)
        #lista_valor_fo.append(fo)
        #print(k*Ts)
        return self.u_prev.tolist(), ref_angle, err_posx, err_posy, err_a, self.int_err_posx, self.int_err_posy, self.int_err_a
    

def filtro_kalman_extendido(x0, P0, sigma_v, Q_k, R_k, lista_entradas, sensor):
    x = x0.reshape((1, len(x0))) # estado real con perturbaciones
    x_k_k = x0.reshape((1, len(x0))) # estado estimado
    x_k1_k = x0.reshape((1, len(x0))) # estado predicho
    z_k = modelo_sensor(x0, lista_entradas[0, :], sensor) # vector de medicion

    P_k_k = P0[np.newaxis,...] # matriz de covarianza del estado estimado
    P_k1_k = P0[np.newaxis,...] # matriz de covarianza del estado predicho

    # calcular iteraciones del filtro
    for k in range(len(t)-1):
        t_aux, x_aux = step_model(modelo_vehiculo, lista_entradas[k,:], 0*sigma_v, t[k], Ts, x[k, :])
        x = np.vstack((x,(x_aux.T)[-1,:])) # medicion real
        # jacobianos del modelo continuo
        A, B, G, C, D, H = jacobianos_vehiculo(x_k_k[k,:], lista_entradas[k,:], sensor)
        # jacobianos modelo discreto
        Ad = expm(A*Ts)
        Bd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(B)
        Gd = ((Ad*Ts).dot(np.eye(len(Ad))-A*Ts/2.)).dot(G)
        A = Ad
        B = Bd
        G = Gd
        # Predicción por integración exacta del modelo linealizado
        t_aux, x_aux = step_model(modelo_vehiculo, lista_entradas[k,:], sigma_v, t[k], Ts, x_k_k[k,:])
        x_k1_k_aux = (x_aux.T)[-1,:]
        x_k1_k = np.vstack((x_k1_k, x_k1_k_aux))                             # Store x_{k|k-1}
        z_k1_k_aux = C.dot(x_k1_k[k+1, :]) + D.dot(lista_entradas[k,:])      # z_{k|k-1}
        #x_k1_k = Ts*modelo_auto(t[k], x[k, :], lista_entradas[k,:], 0*sigma_v) + x_k_k[k, :]
        #z_k1_k_aux = modelo_sensor(x_k1_k[k+1, :], lista_entradas[k,:], 0*sigma_w)
        # prediccion de matriz de covarianza
        P_k1_k_aux = (A.dot(P_k_k[k,:,:])).dot(A.T) + (G.dot(Q_k)).dot(G.T)  # Px_{k|k-1}
        S_k1_k_aux = (C.dot(P_k1_k_aux)).dot(C.T) + (H.dot(R_k)).dot(H.T)    # S_{k|k-1}
        # Actualizacion/correccion del estado y covarianza del proceso
        z_k1_aux = modelo_sensor(x[k+1,:], lista_entradas[k,:], sensor)           # z_{k}
        e = z_k1_aux - z_k1_k_aux                                            # e_{k}
        K = (P_k1_k_aux.dot(C.T)).dot(np.linalg.inv(S_k1_k_aux))             # K_{k}
        x_k_k_aux = x_k1_k[k+1,:] + K.dot(e)                                 # x_{k|k}
        P_k_k_aux = P_k1_k_aux - (K.dot(S_k1_k_aux)).dot(K.T)                # P_{k|k}
        z_k = np.vstack((z_k, z_k1_aux))                                     # Store z_{k}
        x_k_k = np.vstack((x_k_k, x_k_k_aux))                                # Store x_{k|k}
        P_k_k = np.vstack((P_k_k, P_k_k_aux[np.newaxis,...]))                # Store P_{k|k}

    
# graficos
## grafico de posicion x versus y con vector de orientacion
def grafico_xy_orientacion(lista_x, lista_y, lista_ref_x, lista_ref_y):
    x = lista_x
    y = lista_y
    dx = np.diff(x)/max(np.diff(x))
    dy = np.diff(y)/max(np.diff(y))
    origen_x = x[:-1]
    origen_y = y[:-1]
    fig0, ax = plt.subplots()
    fig0.canvas.manager.set_window_title('Trayectoria')
    ax.plot(x, y, 'o-', label='Posición', zorder=1)
    ax.plot(lista_ref_x, lista_ref_y, '--k', label='referencia')
    ax.quiver(origen_x, origen_y, dx, dy, scale=2, scale_units='xy', angles='xy', color='r', label='Vector de dirección', zorder = 5)
    ax.set_xlim(min(x) - 1, max(x) + 1)
    ax.set_ylim(min(y) - 1, max(y) + 1)
    ax.set_title('Trayectoria X versus Y')
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.legend()
    plt.show()
## grafico de torques de entrada para cada actuador
def grafico_torques_entrada(t, lista_entradas):
    fig1, (ax1, ax2) = plt.subplots(2, 1)
    fig1.canvas.manager.set_window_title('Entradas')
    ax1.plot(t, lista_entradas[:, 0], 'r')
    ax1.set_title('Entrada actuador derecho')
    ax1.set_ylabel('Td [Nm]')
    ax2.plot(t, lista_entradas[:, 1], 'b')
    ax2.set_title('Entrada actuador izquierdo')
    ax2.set_xlabel('Tiempo [s]')
    ax2.set_ylabel('Ti [Nm]')
    plt.tight_layout()
    plt.show()
### grafico de variables de estado
def grafico_estado(t, lista_x, lista_ref_x, lista_ref_y, lista_ref_angle):
    titles = ['Posición X [m]', 'Posición Y [m]', 'Ángulo [°]', 'Velocidad angular [rad/s]', 'Velocidad lineal [m/s]', 'Integral error x', 'Integral error y', 'Integral error theta']
    colors = ['r', 'b', 'g', 'm', 'c', 'k', 'r', 'r', 'r']
    for i in range(len(lista_x)):
        fig2, axs = plt.subplots()
        fig2.canvas.manager.set_window_title('Variables de Estado')
        if i == 0:
            axs.plot(t, lista_x[i, :], colors[i])
            axs.plot(t, lista_ref_x, '--k')
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('X [m]')
            pass
        elif i == 1:
            axs.plot(t, lista_x[i, :], colors[i])
            axs.plot(t, lista_ref_y, '--k')
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('Y (m)')
        elif i == 2:
            axs.plot(t, lista_x[i, :]*180/np.pi, colors[i], label='Ángulo [°]')
            axs.plot(t, lista_ref_angle*180/np.pi, 'k--', label='Referencia')
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('theta [°]')
        elif i == 3:
            axs.plot(t, lista_x[i, :], colors[i])
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('V [m/s]')
        elif i == 4:
            axs.plot(t, lista_x[i, :]*180/np.pi, colors[i])
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('w [°/s]')
        elif i == 5:
            axs.plot(t, lista_x[i, :], colors[i])
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('Integral error x')
        elif i == 6:
            axs.plot(t, lista_x[i, :], colors[i])
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('Integral error y')
        elif i == 7:
            axs.plot(t, lista_x[i, :]*180/np.pi, colors[i])
            axs.set_title(titles[i])
            axs.set_xlabel('Tiempo [s]')
            axs.set_ylabel('Integral error theta')
        plt.show()
### perfil de aceleracion
def grafico_aceleracion(t, lista_x):
    velocidad_lineal = lista_x[3, :]
    aceleracion_lineal = np.diff(velocidad_lineal)
    fig2 = plt.figure()
    fig2.canvas.manager.set_window_title('Perfil de aceleración lineal')
    plt.plot(t[:-1], aceleracion_lineal, 'r')
    plt.title('Perfil de aceleración lineal')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('a [m/s^2]')
    plt.show()
    velocidad_angular = lista_x[4, :]
    aceleracion_angular = np.diff(velocidad_angular)
    fig2 = plt.figure()
    fig2.canvas.manager.set_window_title('Perfil de aceleración lineal')
    plt.plot(t[:-1], aceleracion_angular*180/np.pi, 'r')
    plt.title('Perfil de aceleración angular')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('alpha [°/s^2]')
    plt.show()


### perfil de aceleracion
def grafico_error(t, lista_error_x, lista_error_y, lista_error_a):
    fig1 = plt.figure()
    fig1.canvas.manager.set_window_title('Error de posición en x')
    plt.title('Error de posición en x')
    plt.plot(t, lista_error_x, 'r')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('X [m]')
    plt.show()
    fig2 = plt.figure()
    fig2.canvas.manager.set_window_title('Error de posición en y')
    plt.title('Error de posición en y')
    plt.plot(t, lista_error_y, 'r')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Y [m]')
    plt.show()
    fig3 = plt.figure()
    fig3.canvas.manager.set_window_title('Error de posición en theta')
    plt.title('Error de posición en theta')
    plt.plot(t, lista_error_a*180/np.pi, 'r')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('theta [°]')
    plt.show()

def calcular_ITAE(error):
    lista_ITAE = []
    for i in range(len(error)):
        ITAE = np.sum(np.abs(error[:i]) * Ts)
        lista_ITAE.append(ITAE)
    return np.array(lista_ITAE)

def funcion_objetivo(x, u, Q, R):
    print(np.shape(x), np.shape(u), np.shape(Q), np.shape(R))
    J = x.T.dot(Q).dot(x) + u.T.dot(R).dot(u)
    #J = (ref - x.T).dot(Q).dot(ref - x) + u.T.dot(R).dot(u)
    return J

def funcion_objetivo_tiempo(lista_fo):
    fo_sum = []
    for i in range(len(lista_fo)):
        fo_acum = np.sum(lista_fo[:i])
        fo_sum.append(fo_acum)
    return fo_sum

def grafico_simple(x, y, title_x, title_y, title):
    fig1 = plt.figure()
    fig1.canvas.manager.set_window_title(title)
    plt.title(title)
    plt.plot(x, y, 'r')
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.show()