import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


re_list = [0,0,4,4,1,1,3,3,1,1,4,4,0,0]
im_list = [0,0,0,1,1,2,2,3,3,4,4,5,5,0]
t_list = np.linspace(0,1,len(re_list))
def f(t_list,z_list):
    func_list = [0]
    for i in range(2,len(t_list)):
        def f(x,i=np.copy(i)):
            return z_list[i-1] + (z_list[i] - z_list[i-1]) / (t_list[i] - t_list[i-1]) * (x - t_list[i-1])
        func_list.append(f)
        
    return lambda x : np.piecewise(x,[((x > t_list[i-1]) & (x<=t_list[i])) for i in range(1,len(t_list))],func_list)

def f_c(t_list,re_list,im_list):
    f_re = f(t_list,re_list)
    f_im = f(t_list,im_list)
    return lambda x : f_re(x) + 1j * f_im(x)

g = f_c(t_list,re_list,im_list)
t_arr = np.linspace(-1,1,100)
z_arr = g(t_arr)
plt.plot(np.real(z_arr),np.imag(z_arr))
plt.show()

#[point_list[i][1] for i in range(len(point_list))]