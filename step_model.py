import numpy as np
from scipy.integrate import odeint
from scipy.integrate import solve_ivp
from functools import partial 

# t, x = step_model(model,u,sigma_v,t0,step_size,x0)
def step_model(model, u, sigma_v, t0, step_size, x0):

    tfinal=t0 + step_size
    #Nsamples = 10+1
    #Nsamples = 1+1
    #tX = np.linspace(t0, tfinal, Nsamples)

    # solve_ivp
    x = solve_ivp(partial(model, u=u, sigma_v=sigma_v), (t0, tfinal), x0, method='RK45')
    # la siguiente linea es utilizada para el filtro de particulas
    #x = solve_ivp(partial(model, u=u, sigma_v=sigma_v), (t0,tfinal), x0, method='BDF')
    return x.t, x.y
    
def main(): # Test step_model
    # Set the seed of the random number generator to zero
    # to have always the same sequence of random numbers.
    import modelo_vehiculo
    np.random.seed(0)

    t = 0.
    # x = (x=1, y=0, theta=45°, v=100, w=0)
    x = np.array([1., 0., 0*np.pi/4., 0., 0.]) 
    # u = (tr=1, tl=0.1)
    u = np.array([1, 1])
    # sigma_v = 
    sigma_v = 0*np.array([0.1,0.05,0.2,np.pi])
    
    DeltaT = 0.01
    t, x = step_model(modelo_vehiculo.modelo_vehiculo, u, sigma_v, t, DeltaT, x)
    
    print('t:', t)
    print('x:', x[:,-1])
    
if __name__ == '__main__': main()