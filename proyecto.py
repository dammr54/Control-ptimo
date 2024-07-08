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

def init_controller(model,data):
    # inicializacion del controlador
    # esta funcion se ejecuta una sola vez al comienzo
    pass
def controller(model, data):
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
# configurar el controlador
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
