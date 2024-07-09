import mujoco as mj
# Import GLFW (Graphics Library Framework) to create a window and an OpenGL context.
from mujoco.glfw import glfw
import OpenGL.GL as gl
import time
import numpy as np
from funciones import *
from params_vehiculo import *
from step_model import *
from modelo_vehiculo import *
import json
#from mujoco import viewer, renderer -> interaccion y representacion visual
# Import GLFW (Graphics Library Framework) to create a window
# and an OpenGL context.
#import glfw  # conda install conda-forge::glfw
xml_path = "car1.xml" # direccion del modelo
t_sim = 10 # tiempo simulacion
# MuJoCo data structures: modelo, camara, opciones de visualizacion ---
model = mj.MjModel.from_xml_path(xml_path) # MuJoCo model
data = mj.MjData(model) # MuJoCo data
cam = mj.MjvCamera() # Abstract camara
opt = mj.MjvOption() # opciones de visualizacion
data.qpos[0] = 0 # pos x
data.qpos[1] = 0 # pos y
data.qpos[2] = 0 # pos z
data.qpos[3] = 0 # 
data.qpos[4] = 0 # roll
data.qpos[5] = 0 # pitch
data.qpos[6] = 0 # yaw
data.qpos[7] = 0 # 
data.qpos[8] = 0 # 
# Set camera configuration
cam.azimuth = 90 # angulo de vista 
cam.elevation = -11 # elevacion
cam.distance = 7.0 # distancia
cam.lookat = np.array([0.0, 0.0, 0]) # hacia donde mira la camara
# inicializar estructuras de datos de visualización
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000) # tamaño maximo de geometrias
# Actualización cinemática
mj.mj_forward(model, data)

# definición del controlador
tipo_control = 2 # 0: manual, 1: PID, 2: LQI
# manual
Accel_lin_des = 0.0
Accel_ang_des = 0.0
# automatico
Torque_R = 0.0
Torque_L = 0.0

# inicializacion
orientacion = np.zeros(3) # inicial [0, 0, 0]
# inicializa controlador PID
Kpa = 5; Kia = 0; Kda = 5
Kpd = 1; Kid = 0.01; Kdd = 0
estado = [0, 0, 0, 0, 0]
# crear el controlador PID para el angulo y distancia
pid_controller = PIDControl(Kpa, Kia, Kda, Kpd, Kid, Kdd, Ts)

# inicializa controlador LQI
Kpa = 30; Kia = 0*Ts; Kda = 0.5/Ts
Kpd = 2; Kid = 0*Ts; Kdd = Ts
estado_lqi = [0, 0, 0, 0, 0, 0, 0, 0]
u0_lqi = [0, 0, 0, 0, 0]
# crear el controlador LQI para el angulo y distancia
lqi_controller = LQIcontrol(estado_lqi, P0_lqi, u0_lqi, sigma_vx, sigma_wg, Kpa, Kia, Kda, Kpd, Kid, Kdd, Ts)
jacob = 'LQI aumentado'

# almacen de datos
lista_tiempo = []
# PID
lista_senal_control_pid = []
lista_ref_x_pid = []
lista_ref_y_pid = []
lista_ref_a_pid = []
lista_err_x_pid = []
lista_err_y_pid = []
lista_err_a_pid = []
lista_estado_x_pid = []
lista_estado_y_pid = []
lista_estado_a_pid = []
lista_estado_v_pid = []
lista_estado_w_pid = []

# LQI
lista_senal_control_lqi = []
lista_ref_x_lqi = []
lista_ref_y_lqi = []
lista_ref_a_lqi = []
lista_err_x_lqi = []
lista_err_y_lqi = []
lista_err_a_lqi = []
lista_estado_x_lqi = []
lista_estado_y_lqi = []
lista_estado_a_lqi = []
lista_estado_v_lqi = []
lista_estado_w_lqi = []
lista_estado_int_err_x_lqi = []
lista_estado_int_err_y_lqi = []
lista_estado_int_err_a_lqi = []

def controller(model, data):
    global Accel_lin_des, Accel_ang_des, Torque_R, Torque_L, x
    # posicion
    posx = data.qpos[0]
    posy = data.qpos[1]
    posz = data.qpos[2]
    # velocidad
    velx = data.sensor('sensor_vel').data[0]
    vely = data.sensor('sensor_vel').data[1]
    vel = np.sqrt(velx**2 + vely**2)
    velz = data.sensor('sensor_vel').data[2]
    vela_x = data.sensor('sensor_gyro').data[0]
    vela_y = data.sensor('sensor_gyro').data[1]
    vela_z = data.sensor('sensor_gyro').data[2]/25 *2*np.pi # arreglo de escala
    # aceleracion
    acex = data.sensor('sensor_accel').data[0]
    acey = data.sensor('sensor_accel').data[1]
    acez = data.sensor('sensor_accel').data[2]
    # integracion giroscopio
    orientacion[0] += vela_x * Ts
    orientacion[1] += vela_y * Ts
    orientacion[2] += vela_z * Ts
    estado = [posx, posy, orientacion[2], vel, vela_z]
    ref = referencias(data.time)
    if tipo_control == 0: # Manual
        Torque_R = Accel_lin_des/2 + Accel_ang_des/2
        Torque_L = Accel_lin_des/2 - Accel_ang_des/2
        data.ctrl[0] = Torque_R
        data.ctrl[1] = Torque_L
    elif tipo_control == 1: # PID
        senal_control, ref_angle, err_posx, err_posy, err_a = pid_controller.calcular_control(estado, ref)
        data.ctrl[0] = senal_control[0]
        data.ctrl[1] = senal_control[1]
        lista_senal_control_pid.append(senal_control)
        lista_ref_x_pid.append(ref[0])
        lista_ref_y_pid.append(ref[1])
        lista_ref_a_pid.append(ref_angle)
        lista_err_x_pid.append(err_posx)
        lista_err_y_pid.append(err_posy)
        lista_err_a_pid.append(err_a)
        lista_estado_x_pid.append(estado[0])
        lista_estado_y_pid.append(estado[1])
        lista_estado_a_pid.append(estado[2])
        lista_estado_v_pid.append(estado[3])
        lista_estado_w_pid.append(estado[4])
    elif tipo_control == 2: # LQI
        senal_control, ref_angle, err_posx, err_posy, err_a, int_err_posx, int_err_posy, int_err_a  = lqi_controller.calcular_control(estado, ref, jacob)
        data.ctrl[0] = senal_control[0]
        data.ctrl[1] = senal_control[1]
        lista_senal_control_lqi.append(senal_control)
        lista_ref_x_lqi.append(ref[0])
        lista_ref_y_lqi.append(ref[1])
        lista_ref_a_lqi.append(ref_angle)
        lista_err_x_lqi.append(err_posx)
        lista_err_y_lqi.append(err_posy)
        lista_err_a_lqi.append(err_a)
        lista_estado_x_lqi.append(estado[0])
        lista_estado_y_lqi.append(estado[1])
        lista_estado_a_lqi.append(estado[2])
        lista_estado_v_lqi.append(estado[3])
        lista_estado_w_lqi.append(estado[4])
        lista_estado_int_err_x_lqi.append(int_err_posx)
        lista_estado_int_err_y_lqi.append(int_err_posy)
        lista_estado_int_err_a_lqi.append(int_err_a)
    lista_tiempo.append(data.time)


# --- Asignación del manejador del controlador ---
mj.set_mjcb_control(controller)
 
# --- Definición de las funciones manejadoras (handlers) de callbacks de GFLW ---
mouse_button_left   = False
mouse_button_middle = False
mouse_button_right  = False
mouse_lastx = 0
mouse_lasty = 0


def keyboard(window, key, scancode, act, mods):
    global Accel_lin_des, Accel_ang_des
    
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)
        Accel_lin_des = 0
        Accel_ang_des = 0
    
    # https://www.glfw.org/docs/3.3/input_guide.html
    # https://www.glfw.org/docs/3.3/group__keys.html
    if act == glfw.PRESS:
        #print(glfw.LOCK_KEY_MODS)
        #print(glfw.MOD_CAPS_LOCK)
            
        if key == glfw.KEY_M:
            automatic = False
            if (mods == glfw.MOD_SHIFT): # or (mods == (glfw.MOD_CAPS_LOCK | glfw.MOD_SHIFT ))):
                print('Control MANUAL')
            else:
                print('Control manual')
            
            # Set gain of torque actuator at right wheel
            model.actuator_gainprm[0, 0] = 1
            
            # Set gain of torque actuator at left wheel
            model.actuator_gainprm[1, 0] = 1
            
        
        if key == glfw.KEY_A:
            automatic = True
            print('Control automático')
            
            # Set gain of torque actuator at right wheel
            model.actuator_gainprm[0, 0] = 1
            
            # Set gain of torque actuator at left wheel
            model.actuator_gainprm[1, 0] = 1
            
    if act == glfw.REPEAT:

        if key == glfw.KEY_UP:
            print('Up')
            Accel_lin_des += 1
        if key == glfw.KEY_DOWN:
            print('Down')
            Accel_lin_des -= 1
        if key == glfw.KEY_LEFT:
            print('Left')
            Accel_ang_des += 1
        if key == glfw.KEY_RIGHT:
            print('Right')
            Accel_ang_des -= 1
        if key == glfw.KEY_ESCAPE:
            print('Bye!')
            glfw.set_window_should_close(window, True) 
            
def mouse_button(window, button, act, mods):
    global mouse_button_left, mouse_button_middle, mouse_button_right
    
    # The mouse button handler is called whenever a button is pressed or released
    #
    # Update button state
    # Sets the corresponding mouse button to 'True' when it is pressed
    # and sets it to 'False' when it is released.   
    mouse_button_left   = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT  ) == glfw.PRESS)
    mouse_button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    mouse_button_right  = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT ) == glfw.PRESS)

    # Update mouse position
    if mouse_button_left or mouse_button_middle or mouse_button_right:
        print('Mouse button pressed at:  ', glfw.get_cursor_pos(window) )
    else:
        print('Mouse button released at: ', glfw.get_cursor_pos(window) )        


def mouse_scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05*yoffset, scene, cam)
    #print('Mouse scroll')


def mouse_move(window, xpos, ypos):
    global mouse_lastx, mouse_lasty
    global mouse_button_left, mouse_button_middle, mouse_button_right
    
    # Compute mouse displacement, save
    dx = xpos - mouse_lastx
    dy = ypos - mouse_lasty
    mouse_lastx = xpos
    mouse_lasty = ypos

    # No buttons down: nothing to do
    if (not mouse_button_left) and (not mouse_button_middle) and (not mouse_button_right):
        #print('Mouse moved without a button pressed')
        return

    # Get current window size
    width, height = glfw.get_window_size(window)

    # Get shift key state
    PRESS_LEFT_SHIFT  = glfw.get_key(window, glfw.KEY_LEFT_SHIFT)  == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # Determine action based on mouse button
    if mouse_button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif mouse_button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)


# Inicialización del motor gráfico OpenGL via funciones GLFW
def glfw_init():
    width, height = 1280, 720
    window_name = 'Vehiculo de tracción diferencial'
    if not glfw.init():
        print("No se pudo inicializar el contexto OpenGL")
        exit(1)
    
    # Crear una ventana y su contexto OpenGL
    window = glfw.create_window(int(width), int(height), window_name, None, None)
    if not window:
        glfw.terminate()
        print("No se pudo inicializar la ventana GLFW")
        exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1) # Solicitar (activar) v-sync
    # Declarar manejadores de devolución de llamadas de teclado y mouse GLFW
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
    #glfw.set_input_mode(window, glfw.LOCK_KEY_MODS, glfw.FALSE)
    return window
def main():
    # Un bucle de renderizado de gráficos GLFW estándar
    #global model, data, scene, context, opt, cam
    window = glfw_init() # Crear una ventana y su contexto OpenGL
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)    
    while not glfw.window_should_close(window):
        # Obtener y ejecutar eventos
        glfw.poll_events()
        # Actualizar la simulación en un paso
        mj.mj_step(model, data)
        # Get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam,
                               mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)
        # Colorear el fondo del canvas
        #gl.glClearColor(1., 1., 1., 1)
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        # Swappear buffers de despliegue (front) y dibujo (back)
        glfw.swap_buffers(window)
    glfw.terminate()

    # Crear diccionario con datos
    if tipo_control == 0:
        pass
    elif tipo_control == 1:
        datos = {
            'Lista1': lista_tiempo,
            'Lista2': lista_senal_control_pid,
            'Lista3': lista_ref_x_pid,
            'Lista4': lista_ref_y_pid,
            'Lista5': lista_ref_a_pid,
            'Lista6': lista_err_x_pid,
            'Lista7': lista_err_y_pid,
            'Lista8': lista_err_a_pid,
            'Lista9': lista_estado_x_pid,
            'Lista10': lista_estado_y_pid,
            'Lista11': lista_estado_a_pid,
            'Lista12': lista_estado_v_pid,
            'Lista13': lista_estado_w_pid
        }
        # Guardar el diccionario en un archivo JSON
        with open('listas_datos_pid.json', 'w') as file:
            json.dump(datos, file)
    elif tipo_control == 2:
        datos = {
            'Lista1': lista_tiempo,
            'Lista2': lista_senal_control_lqi,
            'Lista3': lista_ref_x_lqi,
            'Lista4': lista_ref_y_lqi,
            'Lista5': lista_ref_a_lqi,
            'Lista6': lista_err_x_lqi,
            'Lista7': lista_err_y_lqi,
            'Lista8': lista_err_a_lqi,
            'Lista9': lista_estado_x_lqi,
            'Lista10': lista_estado_y_lqi,
            'Lista11': lista_estado_a_lqi,
            'Lista12': lista_estado_v_lqi,
            'Lista13': lista_estado_w_lqi,
            'Lista14': lista_estado_int_err_x_lqi,
            'Lista15': lista_estado_int_err_y_lqi,
            'Lista16': lista_estado_int_err_a_lqi
        }
        # Guardar el diccionario en un archivo JSON
        with open('listas_datos_lqi.json', 'w') as file:
            json.dump(datos, file)

if __name__ == '__main__': main()