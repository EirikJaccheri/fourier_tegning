import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

re_list = [0,4,4,1,1,3,3,1,1,4,4,0,0]
im_list = [0,0,1,1,2,2,3,3,4,4,5,5,0]

#status: Fungerer med 2 punkter ikke med 3.

#re_list = [0,3]
#im_list = [0,1]

Z_LIST = np.array(re_list) + 1j * np.array(im_list)


def line_int(z_1, z_2, t_1, t_2, n):
    del_t = t_2 - t_1
    if n == 0:
        return z_1 * (del_t) + (z_2 - z_1) / del_t * (del_t)**2 / 2
    else:
        a = -2*np.pi*1j*n
        int1 = z_1 / a * (np.exp(a*del_t) - 1)
        int2 = (z_2 - z_1) / del_t  * (np.exp(a*del_t) * (a * del_t - 1) + 1) / (a**2)
        return np.exp(a*t_1) * (int1 + int2)
    
def one_coef(z_list, n, del_t):
    coef = 0
    for i in range(1, len(z_list)):
        coef += line_int(z_list[i-1], z_list[i], i*del_t, (i+1)*del_t, n) #FIKS!
    return coef    
	    
    
def fourier_coeff(z_list, num_terms):
    del_t = 1 / (len(z_list - 1))
    c_arr = np.zeros(num_terms, dtype = complex)
    for i in range(num_terms):
        c_arr[i] = one_coef(z_list, i- int(num_terms / 2), del_t)
    return c_arr


def fourier_series(N, num_terms, z_list):
    """
    N : number of time-points, num_terms: number of fourier terms used, ft :  complex 1-periodic function of time to find fourier series from
    """
    c_arr = fourier_coeff(z_list, num_terms)
    print(c_arr)
    four_terms = np.zeros((num_terms, N), dtype= complex)
    t = np.linspace(0, 1, N)
    
    for n in range(int(-num_terms / 2) , int(num_terms / 2) + 1):
        four_terms[int(n + num_terms / 2)] = c_arr[int(n + num_terms / 2)] * np.exp(1j * 2 * np.pi * t * n)
    return  t, four_terms



def init():  
    """
    Initiates lines and circle objects
    """
    line.set_data([], [])
    line_2.set_data([], [])

    for j in range(0, 2*draw_num + 1):
        ax.add_patch(circles[j])
    return [line] +  [line_2] +  circles #must return iterable for blitting to work

def animate(i): #i = frame number
    #line from circle to circle, blue line
    if i > 0:  
        xdata[i-1] = 0
        ydata[i-1] = 0 
    
    for itr, j in enumerate(range(1, draw_num+ 1)):
        xdata[i + j + itr -1 ] = np.real(np.sum(four_terms[(mid-j +1):mid+ j, i]))
        ydata[i + j + itr -1] = np.imag(np.sum(four_terms[(mid-j +1):mid+ j, i])) 

        xdata[i + j  + itr] = np.real(np.sum(four_terms[mid-j:mid+ j, i]) )
        ydata[i + j  + itr] = np.imag(np.sum(four_terms[mid-j:mid+ j, i]) )
        
    xdata[i+2*draw_num ] = np.real(np.sum(four_terms[(mid- draw_num) :mid+ draw_num + 1, i]))
    ydata[i+2*draw_num ] = np.imag(np.sum(four_terms[(mid- draw_num):mid+ draw_num + 1, i]))
    line.set_data(xdata[:i+2*draw_num  + 1], ydata[:i+2*draw_num  + 1 ])

    #red drawn line
    xdata_2[i] = np.real(np.sum(four_terms[(mid- draw_num) :mid+ draw_num + 1 , i]))
    ydata_2[i] = np.imag(np.sum(four_terms[(mid- draw_num) :mid+ draw_num  + 1, i]))
    line_2.set_data(xdata_2[:i+1], ydata_2[:i+1])


    # update center of circles
    for itr, j in enumerate(range(1, draw_num + 1)):
        xc = np.real(np.sum(four_terms[(mid-j +1):mid+ j , i]))
        yc = np.imag(np.sum(four_terms[(mid-j +1):mid+ j , i]))
        circles[j + itr].center = (xc , yc)

        xc2 = np.real(np.sum(four_terms[mid-j:mid+ j , i]) )
        yc2 = np.imag(np.sum(four_terms[mid-j:mid+ j , i]) )

        circles[j+1 + itr].center = (xc2,yc2)

    return [line] +  [line_2] +  circles #must return iterable for blitting to work

def make_animation_objects(N):
    """
    N : number of t-values and frames
    Creates lines, circles and fig, ax objects. 
    """
    plt.style.use('dark_background') 
    fig = plt.figure( figsize = (10, 10))

    ax = plt.axes(xlim = (-6, 6), ylim = (-6 , 6))
    plt.axis("off")

    line, = ax.plot([], [], lw = 0.5, color = "c") 
    line_2, = ax.plot([], [], lw = 2, color = "r") 

    circles = []
    circles.append(plt.Circle((0, 0), np.absolute((four_terms[mid, 0])), fill = False, color = 'c', lw = 0.5))
    for i in range(1, draw_num+ 1):
        circles.append(plt.Circle((0, 0), np.absolute((four_terms[mid- i, 0])), fill = False, color = 'c', lw = 0.5))
        circles.append(plt.Circle((0, 0), np.absolute((four_terms[mid+ i, 0])), fill = False, color = 'c', lw = 0.5))

    xdata, ydata = np.zeros(N + 2*draw_num + 1 ), np.zeros(N+ 2*draw_num + 1)  
    xdata_2, ydata_2 = np.zeros(N+ draw_num+ 1 ), np.zeros(N+ draw_num+ 1)
    return fig, ax, line, line_2, circles, xdata, ydata, xdata_2, ydata_2

def make_params(N, num_terms, z_list, interval):
    #NB: num_terms must be odd, to get same number of positive as negative frquencies
    draw_num = int((num_terms - 1) / 2)
    t, four_terms = fourier_series(N, num_terms, z_list)
    mid = int(num_terms / 2) #constant term
    return num_terms, draw_num, N, t, four_terms, mid, interval

#Change only input parameters here
num_terms, draw_num, N, t, four_terms, mid, interval = make_params(N = 200, num_terms = 31, z_list=Z_LIST, interval = 20) #N: number of t-points/frames. Animation slows down when N increases.
#interval: ms between each animation


fig, ax, line, line_2, circles, xdata, ydata, xdata_2, ydata_2 = make_animation_objects(N)
anim = animation.FuncAnimation(fig, animate, init_func= init, frames = N, interval = interval,  blit = True) 
plt.draw()
plt.show()
