import math
import numpy as np

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
        return senal_control, ref_angle, err_d, err_a
    

def referencias(t, t1=1.05, t2=5, t3=10, t4=15):
    if t < t1:
        ref = [1, 1]
        #ref = [0, 2]
        ref = [1, 1] # -
    elif t >= t1 and t < t2:
        ref = [-1, 1]
        ref = [1, 1] # -
    elif t >= t2 and t < t3:
        ref = [-1, -1]
        #ref = [2, 0]
        ref = [1, 1] # -
    elif t >= t3 and t < t4:
        ref = [1, -1]
        ref = [1, 1] # -
    elif t >= t4:
        ref = [1, 1]
        ref = [1, 1] # -
    return ref