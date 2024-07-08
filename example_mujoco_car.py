# Added GLFW keyboard and mouse handlers following the example in:
# https://github.com/tayalmanan28/MuJoCo-Tutorial/blob/main/mujoco_base.py
#
import mujoco # Note does not need mujoco.viewer like the other examples
#from mujoco import viewer, renderer

# Import GLFW (Graphics Library Framework) to create a window
# and an OpenGL context.
#import glfw  # conda install conda-forge::glfw
from mujoco.glfw import glfw # MuJoCo 3.1.1 includes glfw
import OpenGL.GL as gl

import time
import numpy as np


# --- Definición de un modelo ---

xml_path = "car1.xml"

# --- MuJoCo data structures: modelo, camara, opciones de visualizacion ---

model = mujoco.MjModel.from_xml_path(xml_path) # MuJoCo model
data = mujoco.MjData(model)                 # MuJoCo data
cam = mujoco.MjvCamera()                    # Abstract camera
opt = mujoco.MjvOption()                    # Visualization options
#print(model.opt.timestep)
#model.opt.timestep = 0.001


data.qpos[0] =  -0.5
data.qpos[1] =   1.0

# Set camera configuration
cam.azimuth = 89.608063
cam.elevation = -11.588379
cam.distance = 5.0
cam.lookat = np.array([0.0, 0.0, 1.5])

# Initialize visualization data structures
#mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)
scene = mujoco.MjvScene(model, maxgeom=10000)
        
# Actualización cinemática
#mujoco.mj_kinematics(model, data)
mujoco.mj_forward(model, data)


# --- Definición del controlador ---

automatic = False
Accel_lin_des = 0.0
Accel_ang_des = 0.0
Torque_R = 0.0
Torque_L = 0.0

def controller(model, data):
    global Accel_lin_des, Accel_ang_des, Torque_R, Torque_L
    
    """
    m*accel = (1/r)*(Tau_R+Tau_L) - c*v
    J*alpha = (W/(2*r))*(Tau_R-Tau_L) - b*omega
    
    accel = [a  a][Tau_R Tau_L]' con a = 1/(m*r)
    alpha = [b -b][Tau_R Tau_L]' con b = W/(2*J*r)

    Tau_R = [1/(2a)  1/(2b)][accel alpha]' = [mr/2  Jr/W][accel alpha]'
    Tau_L = [1/(2a) -1/(2b)][accel alpha]' = [mr/2 -Jr/W][accel alpha]' 
    """
    
    if automatic: 
        # """
        # Place you controller here.
        # """
        Torque_R = Accel_lin_des/2 + Accel_ang_des/2
        Torque_L = Accel_lin_des/2 - Accel_ang_des/2
        data.ctrl[0] = Torque_R #data.qpos[0] + dq[0, 0] # q1 position servo 1
        data.ctrl[1] = Torque_L #data.qpos[1] + dq[1, 0] # q2 position servo 2
    
    else: # Manual
        Torque_R = Accel_lin_des/2 + Accel_ang_des/2
        Torque_L = Accel_lin_des/2 - Accel_ang_des/2
        data.ctrl[0] = Torque_R
        data.ctrl[1] = Torque_L
        
    
    return None

# --- Asignación del manejador del controlador ---
mujoco.set_mjcb_control(controller)
 
# --- Definición de las funciones manejadoras (handlers) de callbacks de GFLW ---
mouse_button_left   = False
mouse_button_middle = False
mouse_button_right  = False
mouse_lastx = 0
mouse_lasty = 0


def keyboard(window, key, scancode, act, mods):
    global Accel_lin_des, Accel_ang_des
    
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
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
            Accel_lin_des += 10
        if key == glfw.KEY_DOWN:
            print('Down')
            Accel_lin_des -= 10
        if key == glfw.KEY_LEFT:
            print('Left')
            Accel_ang_des += 10
        if key == glfw.KEY_RIGHT:
            print('Right')
            Accel_ang_des -= 10
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
    action = mujoco.mjtMouse.mjMOUSE_ZOOM
    mujoco.mjv_moveCamera(model, action, 0.0, -0.05*yoffset, scene, cam)
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
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif mouse_button_left:
        if mod_shift:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)


# --- Inicialización del motor gráfico OpenGL via funciones GLFW ---
def glfw_init():
    width, height = 1280, 720
    window_name = 'Ejemplo de ventana GLFW'
    if not glfw.init():
        print("Could not initialize OpenGL context")
        exit(1)
    
    # Do not use these lines... pyopengl from Anaconda os 3.1.1
    # since Nov. 2023 and at list today (2024.03.09). 
    # OS X supports only forward-compatible core profiles from 3.2
    #glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    #glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    #glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    #glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, gl.GL_TRUE)
    
    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window( int(width), int(height), window_name, None, None)
        
    if not window:
        glfw.terminate()
        print("Could not initialize GLFW window")
        exit(1)
    
    glfw.make_context_current(window)
    glfw.swap_interval(1) # Request (activate) v-sync
                          # https://discourse.glfw.org/t/newbie-questions-trying-to-understand-glfwswapinterval/1287
 

    # Declare GLFW mouse and keyboard callback handlers
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    glfw.set_scroll_callback(window, mouse_scroll)
 
    #glfw.set_input_mode(window, glfw.LOCK_KEY_MODS, glfw.FALSE)
    
    return window
    
# A standard GLFW graphics rendering loop has the following structure.
# See example at: https://pypi.org/project/glfw/
#
# window = glfw_init() # Create a window and its OpenGL context
# 
# while not glfw.window_should_close(window):
#     # Poll for and process events
#     glfw.poll_events()
#
#     # Render someting here, e.g. using pyOpenGL
#     gl.glClearColor(1., 1., 1., 1)
#     gl.glClear(gl.GL_COLOR_BUFFER_BIT)
#
#     # Swap front and back buffers
#     glfw.swap_buffers(window)
#
# glfw.terminate() 
#

def main():
    #global model, data, scene, context, opt, cam

    window = glfw_init()
    
    # WARNING: MuJoCo OpenGL context cannot be created using MjrContext
    #          before creating a window with glfw or OpenGL.
    # Therefore the following function cannot be invoked before the 
    # main(). However, note that it is possible to create an MjvScene 
    # before creating a window with glfw and assigning an OpenGL context.
    # In fact, the "scene" with all geometrical objects can exist without
    # ever "painting" (rendering) the scene onto a given context.
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)    
    
    # Loop de visualización GLFW
    # Ver ejemplos básicos en:
    # https://codeloop.org/python-modern-opengl-glfw-window/
    # https://www.programcreek.com/python/example/124904/glfw.window_should_close
    # https://medium.com/@shashankdwivedi6386/pyopengl-creating-simple-window-in-python-9ae3b10f6355
    # https://github.com/tayalmanan28/MuJoCo-Tutorial/blob/main/examples/control_pendulum.py
    while not glfw.window_should_close(window):
        # Obtener y ejecutar eventos
        glfw.poll_events()
                
        # Actualizar la simulación en un paso
        mujoco.mj_step(model, data)
        
        # Obtenemos algunos datos de sensores
        # Notar que algunos datos se pueden obtener directamente de variables
        # de estado como qpos, qvel sin necesidad de crear un sensor
        # Ver: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
        print('Position-site_xpos: ', data.site_xpos[0])
        print('Position-qpos: ', data.qpos[0], data.qpos[1], data.qpos[2]) 
        print('Velocity-sensor: ', data.sensor('sensor_vel').data)
        print('Velocity-qvel  : ', data.qvel[0], data.qvel[1], data.qvel[2],)
        
        print('Acceleration-sensor: ', data.sensor('sensor_accel').data)
        print('Acceleration-qacc  : ', data.qacc[0], data.qacc[1], data.qacc[2])
        print('Gyro: ', data.sensor('sensor_gyro').data)

        #print('Sensor data: ', data.sensordata)
        
        
        # Get framebuffer viewport
        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, viewport_width, viewport_height)

        # Update scene and render
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL.value, scene)
        mujoco.mjr_render(viewport, scene, context)

        # Colorear el fondo del canvas
        #gl.glClearColor(1., 1., 1., 1)
        #gl.glClear(gl.GL_COLOR_BUFFER_BIT)

        # Swappear buffers de despliegue (front) y dibujo (back)
        glfw.swap_buffers(window)
 
    glfw.terminate()
 
 
if __name__ == '__main__': main()