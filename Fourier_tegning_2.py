import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation

### Eirik sin kode:
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
### hans sin kode:

def one_coeff(ft, term_num):
    def real_func(x):
        return np.real(ft(x)* np.exp(-1j * 2 * np.pi * term_num*x))

    def imag_func(x):
        return np.imag(ft(x)*np.exp(-1j * 2 * np.pi * term_num*x))

    real_integral = integrate.quad(real_func, 0, 1)
    imag_integral = integrate.quad(imag_func, 0, 1)

    return real_integral[0] + 1j*imag_integral[0]
    
def fourier_coeff(ft, num_terms):
    c_arr = np.zeros(num_terms, dtype = complex)
    for i in range(int(-num_terms / 2)-1, int(num_terms / 2)+ 1):
        c_arr[int(i + num_terms / 2)] = one_coeff(ft, i)
    return c_arr 

def make_spin(N, num_terms, ft):
    c_arr = fourier_coeff(ft, num_terms)
    four_terms = np.zeros((num_terms, N), dtype= complex)
    t = np.linspace(0, 1, N)
    four = 0
    for n in range(int(-num_terms / 2) -1, int(num_terms / 2) + 1):
        four_terms[int(n + num_terms / 2)] = c_arr[int(n + num_terms / 2)] * np.exp(1j * 2 * np.pi * t * n) 
        four += c_arr[int(n + num_terms / 2) ] * np.exp(1j * 2 * np.pi * t * n) 

    return four, t, four_terms


def ft(x):
    return g(x)

num_terms = 35 #Velg num_terms som oddetall midlertidig
draw_num = int((num_terms - 1) / 2)
assert draw_num < num_terms
N = 500 #antall t-verdier / frames
four, t, four_terms = make_spin(N, num_terms, ft)
k = int(num_terms / 2) #jalla, skal fikse



def init(): #must use name init. Initiates line and circle.
    line.set_data([], [])
    line2.set_data([], [])
    #line3.set_data([], [])

    for j in range(0, 2*draw_num + 1):
        ax.add_patch(circles[j])
    return [line] +  [line2] +  circles #+ [line3]

def animate(i): #i = frame number
    #line from circle to circle, blue line
    if i > 0:  
        xdata[i-1] = 0
        ydata[i-1] = 0 
    xdata[i] = np.real(four_terms[k, i])
    ydata[i] = np.imag(four_terms[k, i])
    itr = 0
    for j in range(1, draw_num+1):
        xdata[i + j + itr] = np.real(np.sum(four_terms[(k-j +1):k + j+ 1, i]))
        ydata[i + j + itr] = np.imag(np.sum(four_terms[(k-j +1):k + j+ 1, i])) 

        xdata[i + (j + 1) + itr] = np.real(np.sum(four_terms[k -j:k + j+ 1, i]) )
        ydata[i +  (j + 1) + itr] = np.imag(np.sum(four_terms[k -j:k + j+ 1, i]) )
        itr += 1

    xdata[i+draw_num + 2] = np.real(four[i])
    ydata[i+draw_num + 2] = np.imag(four[i])
    line.set_data(xdata[:i+draw_num + 3], ydata[:i+draw_num + 3 ])

    #red drawn line, should match blue line..
    # xdata_test[i] = np.real(np.sum(four_terms[(k - draw_num):k + draw_num + 1, i]))
    # ydata_test[i] = np.imag(np.sum(four_terms[(k - draw_num):k + draw_num + 1, i]))
    # line3.set_data(xdata_test[:i+1], ydata_test[:i+1])

    #the figure, red line
    xdata_sum[i] = np.real(four[i])
    ydata_sum[i] = np.imag(four[i])
    line2.set_data(xdata_sum[:i+1], ydata_sum[:i+1]) 
    
    #center of circles update
    itr = 0
    for j in range(1, draw_num + 1):
        xc = np.real(np.sum(four_terms[(k-j +1):k + j+ 1, i]))
        yc = np.imag(np.sum(four_terms[(k-j +1):k + j+ 1, i]))
        circles[j + itr].center = (xc , yc)

        xc2 = np.real(np.sum(four_terms[k -j:k + j+ 1, i]) )
        yc2 = np.imag(np.sum(four_terms[k -j:k + j+ 1, i]) )

        circles[j+1 + itr].center = (xc2,yc2)
        itr += 1

    return [line] +  [line2] +  circles #+ [line3]

plt.style.use('dark_background') #must stand before fig,ax
fig = plt.figure( figsize = (10, 10))
ax = plt.axes(xlim = (-6, 6), ylim = (-6 , 6))
line, = ax.plot([], [], lw = 0.5, color = "c") 
line2, = ax.plot([], [], lw = 2, color = "r") #lw = linewidth, r = red, c = cyan
#line3, = ax.plot([], [], lw = 2, color = "r") 
circles = []
circles.append(plt.Circle((0, 0), np.absolute((four_terms[k, 0])), fill = False, color = 'c', lw = 0.5))
for i in range(1, draw_num+ 1):
    circles.append(plt.Circle((0, 0), np.absolute((four_terms[k + i, 0])), fill = False, color = 'c', lw = 0.5))
    circles.append(plt.Circle((0, 0), np.absolute((four_terms[k - i, 0])), fill = False, color = 'c', lw = 0.5))
frames = N
xdata, ydata = np.zeros(frames + 2*draw_num + 1 ), np.zeros(frames+ 2*draw_num + 1)  #reason for (x,y) = (0,0) always drawn
xdata_sum, ydata_sum = np.zeros(frames+ draw_num+ 1 ), np.zeros(frames+ draw_num+ 1)
xdata_test, ydata_test = np.zeros(frames+ draw_num+ 1 ), np.zeros(frames+ draw_num+ 1)
plt.axis("off")
anim = animation.FuncAnimation(fig, animate, init_func= init, frames = frames, interval = 1,  blit = True) 
plt.draw()
plt.show()