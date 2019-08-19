# Assignment 9 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 6/4/2019
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 9,6
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 18
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"
'''
Function to select different functions
Arguments:
 t -> vector of time values
 n -> encoded from 1 to 3 to select function
 w -> frequency of cos^3 function, which is by default 0.86
'''

def f(t,n,w=None,d=None):
    if(n == 1):
        return sin(sqrt(2)*t)
    elif(n==2):
        if(w is None):
            return pow(cos(0.86*t),3)
        elif(w!=None):
            return pow(cos(w*t),3)
    elif(n==3):
        return cos(16*(1.5+t/(2*pi))*t)
    elif(n==4):
        return t
    elif(n==5):
        if(w is None):
            return cos(0.86*t)
        elif(w!=None and d!=None):
            return cos(w*t+d)
    else:
        return sin(sqrt(2)*t)   
def window_fn(n,N):
    return (0.54+0.46*cos(2*pi*n/N))
'''
Function to find Discrete Fourier Transform
Arguments:
 low_lim,up_lim -> lower & upper limit for time vector
 no_points      -> Sampling rate
 f              -> function to compute DFT for
 n              -> mapped value for a function ranges(1,6)
 norm_factor    -> default none, only for Gaussian function
                   it is given as parameter
'''

def findFFT(low_lim,up_lim,no_points,n,window=True,wo=None,d=None,eps=0):
    t = linspace(low_lim,up_lim,no_points+1)[:-1]
    dt=t[1]-t[0]
    fmax=1/dt
    N = no_points
    y = f(t,n,wo,d)+eps*randn(len(t))
    
    if(window):
        n1=arange(N)
        wnd=fftshift(window_fn(n1,N))
        y=y*wnd
        
    y[0]=0        # the sample corresponding to -tmax should be set zeroo
    y=fftshift(y) # make y start with y(t=0)
    Y=fftshift(fft(y))/N

    w = linspace(-pi*fmax,pi*fmax,N+1)[:-1]        
    return t,Y,w
'''
Function to plot Magnitude and Phase spectrum for given function
Arguments:
 t              -> time vector
 Y              -> DFT computed
 w              -> frequency vector
 Xlims,Ylims    -> limits for x&y axis for spectrum
 plot_title,fig_no -> title of plot and figure no
'''

def plot_FFT(t,Y,w,Xlims,plot_title,fig_no,dotted=False,Ylims=None):
    
    figure()
    subplot(2,1,1)
    
    if(dotted):
        plot(w,abs(Y),'b',w,abs(Y),'bo',lw=2)
    else:
        plot(w,abs(Y),'b',lw=2)

    xlim(Xlims)
        
    ylabel(r"$|Y(\omega)| \to$")
    title(plot_title)
    grid(True)
    
    ax = subplot(2,1,2)
    ii=where(abs(Y)>0.005)
    plot(w[ii],angle(Y[ii]),'go',lw=2)

    if(Ylims!=None):
        ylim(Ylims)
    
    xlim(Xlims)
    ylabel(r"$\angle Y(j\omega) \to$")
    xlabel(r"$\omega \to$")
    grid(True)
    savefig("fig10-"+fig_no+".png")
    show()
t,Y,w = findFFT(-pi,pi,64,1,False)
Xlims = [-10,10]
title_plot = r"Spectrum of $\sin\left(\sqrt{2}t\right)$"
plot_FFT(t,Y,w,Xlims,title_plot,"1")
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]

figure(2)
plot(t1,f(t1,1),'b',lw=2)
plot(t2,f(t2,1),'r',lw=2)
plot(t3,f(t3,1),'r',lw=2)
ylabel(r"$y(t) \to$",size=16)
xlabel(r"$t \to$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$")
grid(True)
savefig("fig10-2.png")
show()
t1=linspace(-pi,pi,65);t1=t1[:-1]
t2=linspace(-3*pi,-pi,65);t2=t2[:-1]
t3=linspace(pi,3*pi,65);t3=t3[:-1]
y=f(t1,1)

figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'go',lw=2)
ylabel(r"$y(t) \to$",size=16)
xlabel(r"$t \to$",size=16)
title(r"$\sin\left(\sqrt{2}t\right)$ with $t$ wrapping every $2\pi$ ")
grid(True)
savefig("fig10-3.png")
show()
t,Y,w = findFFT(-pi,pi,64,4,False)

figure()
semilogx(abs(w),20*log10(abs(Y)),lw=2)
xlim([1,10])
ylim([-20,0])
xticks([1,2,5,10],["1","2","5","10"],size=16)
ylabel(r"$20\log_{10}|Y(\omega)|$ (dB) $\to$ ",size=16)
title(r"Spectrum of a digital ramp")
xlabel(r"$\omega \to$",size=16)
grid(True)
savefig("fig10-4.png")
show()
t1=linspace(-pi,pi,65)[:-1]
t2=linspace(-3*pi,-pi,65)[:-1]
t3=linspace(pi,3*pi,65)[:-1]
n=arange(64)
wnd=fftshift(window_fn(n,64))
y=f(t1,1)*wnd

figure(3)
plot(t1,y,'bo',lw=2)
plot(t2,y,'ro',lw=2)
plot(t3,y,'go',lw=2)
ylabel(r"$y(t) \to$",size=16)
xlabel(r"$t \to$",size=16)
plt_title = r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$"
title(plt_title)
grid(True)
savefig("fig10-5.png")
show()
t,Y,w = findFFT(-pi,pi,64,1,True)
Xlims = [-8,8]
title_plot = r"Spectrum of $\sin\left(\sqrt{2}t\right)\times \omega(t)$"
plot_FFT(t,Y,w,Xlims,title_plot,"6",True)
t,Y,w = findFFT(-4*pi,4*pi,256,1,True)
Xlims = [-4,4]
plot_title = r"Spectrum of $\sin\left(\sqrt{2}t\right)\times \omega(t)$"
plot_FFT(t,Y,w,Xlims,plot_title,"7",True)
t,Y,w = findFFT(-4*pi,4*pi,256,2,True)
Xlims = [-8,8]
plot_title = r"Spectrum of Windowed $\cos^3(\omega_o t)$"
plot_FFT(t,Y,w,Xlims,plot_title,"8")
t,Y,w = findFFT(-4*pi,4*pi,256,2,False)
Xlims = [-8,8]
plot_title = r"Spectrum of $\cos^3(\omega_o t)$ without windowing"
plot_FFT(t,Y,w,Xlims,plot_title,"9")
def estimate_omega(low_lim,up_lim,eps):
    w_actual = np.random.uniform(low_lim,up_lim)
    delta_actual = (randn())
    t,Y,w = findFFT(-1*pi,1*pi,128,5,True,w_actual,delta_actual,eps=eps)
    
    Y_half = Y[int(len(Y)/2):]
    w_half = w[int(len(w)/2):]
    k = 0.1
    idx = np.where(abs(Y_half) >= np.mean(abs(Y_half))+k*sqrt(np.var(abs(Y_half))))
    
    w0 = np.matmul(w_half[idx],
                   np.transpose(abs(Y_half[idx])))/(np.sum(abs(Y_half[idx])))
    
    w_peak_idx = (np.abs(w_half-w0)).argmin()
    delta      = angle(Y_half[w_peak_idx])
    
    print("Actual w0 : %g , Actual delta : %g"%(w_actual,delta_actual))
#     Xlims = [-8,8]
#     plot_title = r"Figure 10 : Spectrum of $\cos(\omega_o t + \delta)$"
#     plot_FFT(t,Y,w,Xlims,plot_title,"10") 
    return t,w0,delta,w_actual,delta_actual
#function to create Matrix for finding the Best fit using lstsq
# with no_of rows, columns by default 2 and vector x as arguments
def createAmatrix(nrow,t,model,wo):
    A = zeros((nrow,2)) # allocate space for A
    A[:,0],A[:,1] = model(t,wo)
    return A
# function to calculate model A
def modelA(t,wo):
    return (cos(wo*t),sin(wo*t))
def estimate_delta(t,wo,y):
    M = createAmatrix(len(y),t,modelA,wo)
    c = (lstsq(M,y)[0])
    #calculating delta
    delta = arccos(c[0]/sqrt(pow(c[0],2)+pow(c[1],2)))
    return delta
def estimator(N,eps):
    est_err_w = []
    est_err_delta = []
    for i in range(N):
        t,wo_est,delta_est,w_actual,delta_actual = estimate_omega(0.5,1.5,eps)
#         actual_fn = cos(w_actual*t+delta_actual)
#         est_fn    = cos(wo_est*t+delta_est)
        est_err_w.append(abs(amax(w_actual-wo_est)))
        est_err_delta.append(abs(amax(delta_actual-delta_est)))
        
        print("Estimated w0 : %g and delta : %g \n"%((wo_est),delta_est))
    return np.mean(est_err_w),np.mean(est_err_delta)
eps = 0
N = 5
estimated_error_w,estimated_error_delta = estimator(N,eps)
print("\nEstimated Error for %g sample signals without Noise addition in w0 and delta : %g , %g"%
      (N,estimated_error_w,estimated_error_delta))
eps = 0.1
N   = 5
estimated_error_w,estimated_error_delta = estimator(N,eps)
print("\nEstimated Error for %g sample signals with Noise addition in w0 and delta : %g , %g"%
      (N,estimated_error_w,estimated_error_delta))
def chirp(t):
    return cos(16*(1.5+t/(2*pi))*t)
t = linspace(-pi,pi,1025)[:-1]
dt=t[1]-t[0]
fmax=1/dt
N = 1024
y = chirp(t)
Y=fftshift(fft(y))/N
w = linspace(-pi*fmax,pi*fmax,N+1)[:-1]       
Xlims = [-100,100]
plot_title = r"Spectrum of Non-Windowed Chirped Signal"
plot_FFT(t,Y,w,Xlims,plot_title,"10")
t = linspace(-pi,pi,1025)[:-1]
dt=t[1]-t[0]
fmax=1/dt
N = 1024
y = chirp(t)

n1=arange(N)
wnd=fftshift(window_fn(n1,N))
y=y*wnd
y[0] = 0
y = fftshift(y)
Y = fftshift(fft(y))/1024.0
w = linspace(-pi*fmax,pi*fmax,1025)[:-1]       

Xlims = [-100,100]
plot_title = r"Spectrum of Windowed Chirped Signal"
plot_FFT(t,Y,w,Xlims,plot_title,"11")
def partition(t,n):
    t_batches = [t[i:n+i] for i in range(len(t)-n)]
    return t_batches
def STFT(t,no_samples,n):
    dt=t[1]-t[0]
#     print(dt)
    fmax=1/dt
    N = no_samples
    y = f(t,n)
    
    n1=arange(N)
    wnd=fftshift(window_fn(n1,N))
    y=y*wnd
        
    Y=fftshift(fft(y))/N
    
    w = linspace(-pi*fmax,pi*fmax,N+1)[:-1]        
    return t,Y,w
n = 64
t_batches = partition(t,n)

batch_dfts = []
batch_ts   = []

for i in range(len(t_batches)):
    t,Y,w = STFT(t_batches[i],n,3)
    batch_dfts.append(Y)
    batch_ts.append(t)
t = linspace(-pi,pi,1025)[:-1]
T, W = np.meshgrid(t[:960],w)
Z = abs(np.array(batch_dfts))
fig = figure()
ax = fig.add_subplot(111)
ax.contourf(T,W,Z.T,cmap='jet')
title("Contour plot of the magnitude of the spectrum")
xlabel(r"$time \to$")
ylabel(r"$\omega \to$")
plt.savefig("fig10-12.png")
show()
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(T,W,(Z.T), cmap=cm.jet,
                       linewidth=0.1)

title(r"Surface plot of $|Y(\omega)|$")
ax.set_xlabel(r'$t \to$')
ax.set_ylabel(r'$\omega \to$')
ax.set_zlabel(r'$|Y(\omega)| \to$')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig("fig10-13.png")
plt.show()