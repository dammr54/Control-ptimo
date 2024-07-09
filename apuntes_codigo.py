import mujoco as mj
# Import GLFW (Graphics Library Framework) to create a window and an OpenGL context.
from mujoco.glfw import glfw
import OpenGL.GL as gl
import time
import numpy as np
xml_path = "car1.xml" # direccion del modelo
t_sim = 10 # tiempo simulacion
# MuJoCo data structures: modelo, camara, opciones de visualizacion ---
model = mj.MjModel.from_xml_path(xml_path) # MuJoCo model
data = mj.MjData(model) # MuJoCo data
cam = mj.MjvCamera() # Abstract camara
opt = mj.MjvOption() # opciones de visualizacion

#print(model.opt.timestep)
#model.opt.timestep = 0.001
#mujoco.mjv_defaultCamera(cam)
#mujoco.mj_kinematics(model, data)

def init_controller(model, data):
    # inicializacion del controlador
    # esta funcion se ejecuta una sola vez al comienzo
    pass
def controller(model, data):
    print(data)
    # controlador
    # esta funcion se llama iterativamente dentro de la simulacion
    pass
def keyboard(window, key, scancode, act, mods):
    # eventos del teclado
    pass
def mouse_button(window, button, act, mods):
    # eventos del mouse
    pass
def mouse_move(window, xpos, ypos):
    # movimientos del mouse
    pass
def scroll(window, xoffset, yoffset):
    pass

# inicializar GLFW (Graphics Library Framework), crear ventana
# make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)
# inicializar visualizacion del data estructure del modelo de mujoco
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
# setear llamadas GLFW de mouse y teclado
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

# inicializar el controlador
init_controller(model, data)
# Asignación del manejador del controlador
mj.set_mjcb_control(controller)

while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)
    if (data.time>=t_sim):
        break
    # obtener la ventana gráfica del framebuffer
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    # Actualizar escena y renderizar
    mj.mjv_updateScene(model, data, opt, None, cam,
    mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)
    # swap OpenGL buffers (blocking call due to v-sync)
    glfw.swap_buffers(window)
    # procesar eventos GUI pendientes, call GLFW callbacks
    glfw.poll_events()
    glfw.terminate()



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
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)    
    
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
        mj.mj_step(model, data)
        
        # Obtenemos algunos datos de sensores
        # Notar que algunos datos se pueden obtener directamente de variables
        # de estado como qpos, qvel sin necesidad de crear un sensor
        # Ver: https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjdata.h
        ## ------------------------------------------------------------
        # posicion de los sites
        #print('Position-site_xpos: ', data.site_xpos[0])
        # posicion sensor framepos
        #print('Position-qpos: ', data.qpos[0], data.qpos[1], data.qpos[2]) 
        # sensor de velocidad
        #print('Velocity-sensor: ', data.sensor('sensor_vel').data)
        #print('Velocity-qvel  : ', data.qvel[0], data.qvel[1], data.qvel[2],)
        # acelerometro
        #print('Acceleration-sensor: ', data.sensor('sensor_accel').data)
        #print('Acceleration-qacc  : ', data.qacc[0], data.qacc[1], data.qacc[2])
        # giroscopio
        #print('Gyro: ', data.sensor('sensor_gyro').data)
        #print(data.qpos)
        ## ------------------------------------------------------------
        # todos los sensores
        #print('Sensor data: ', data.sensordata)
        
        
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
 
 
if __name__ == '__main__': main()