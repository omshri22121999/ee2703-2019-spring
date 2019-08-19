# Assignment 7 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 21/3/2019
from pylab import *


plt.rcParams['savefig.dpi'] = 75

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 12, 9
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
plt.rcParams['text.latex.preamble'] = r"\usepackage{subdepth}, \usepackage{type1cm}"

'''
Function to select different functions
Arguments:
 t -> vector of time values
 n -> encoded from 1 to 6 to select function
'''


def func_select(t, n):
    if(n == 1):
        return sin(5*t)
    elif(n == 2):
        return (1+0.1*cos(t))*cos(10*t)
    elif(n == 3):
        return pow(sin(t), 3)
    elif(n == 4):
        return pow(cos(t), 3)
    elif(n == 5):
        return cos(20*t + 5*cos(t))
    elif(n == 6):
        return exp(-pow(t, 2)/2)
    else:
        return sin(5*t)


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


def findFFT(low_lim, up_lim, no_points, f, n, norm_Factor=None):
    t = linspace(low_lim, up_lim, no_points+1)[:-1]
    y = func_select(t, n)
    N = no_points

    # DFT for gaussian function
    # ifftshift is used to center the function to zero
    # norm_factor is multiplying constant to DFT

    if(norm_Factor != None):
        Y = fftshift((fft(ifftshift(y)))*norm_Factor)
    else:
        # normal DFT for periodic functions
        Y = fftshift(fft(y))/(N)

    w_lim = (2*pi*N/((up_lim-low_lim)))
    w = linspace(-(w_lim/2), (w_lim/2), (no_points+1))[:-1]
    return t, Y, w


'''
Function to plot Magnitude and Phase spectrum for given function
Arguments:
 t              -> time vector
 Y              -> DFT computed
 w              -> frequency vector
 threshold      -> value above which phase is made zero
 Xlims,Ylims    -> limits for x&y axis for spectrum
 plot_title,fig_no -> title of plot and figure no
'''


def plot_FFT(t, Y, w, threshold, Xlims, plot_title, fig_no, Ylims=None):

    subplot(2, 1, 1)
    plot(w, abs(Y), lw=2)
    xlim(Xlims)
    if(Ylims != None):
        ylim(Ylims)

    ylabel(r"$|Y(\omega)| \to$")
    title(plot_title)
    grid(True)

    ax = subplot(2, 1, 2)
    ii = where(abs(Y) > threshold)
    plot(w[ii], angle(Y[ii]), 'go', lw=2)

    if(Ylims != None):
        ylim(Ylims)

    xlim(Xlims)
    ylabel(r"$\angle Y(j\omega) \to$")
    xlabel(r"$\omega \to$")
    grid(True)
    show()


'''
DFT for sin(5t) computed in incorrect way
* like without normalizing  factor
* without centering fft of function to zero
'''

x = linspace(0, 2*pi, 128)
y = sin(5*x)
Y = fft(y)
subplot(2, 1, 1)
plot(abs(Y), lw=2)
title(r"Figure 1 : Incorrect Spectrum of $\sin(5t)$")
ylabel("$|Y(\omega)|$")
grid(True)
subplot(2, 1, 2)
plot(unwrap(angle(Y)), lw=2)
xlabel(r"$\omega \to $")
ylabel(r"$\angle Y(\omega)$")
grid(True)
show()
t, Y, w = findFFT(0, 2*pi, 128, f, 1)
Xlims = [-15, 15]
plot_FFT(t, Y, w, 1e-3, Xlims, r"Figure 2: Spectrum of $\sin(5t)$", "2")
t, Y, w = findFFT(0, 2*pi, 128, f, 2)
Xlims = [-15, 15]
Ylims = []
plot_FFT(t, Y, w, 1e-4, Xlims,
         r"Figure 3: Incorrect Spectrum of $(1+0.1\cos(t))\cos(10t)$", "3")
t, Y, w = findFFT(-4*pi, 4*pi, 512, f, 2)
Xlims = [-15, 15]
Ylims = []
plot_FFT(t, Y, w, 1e-4, Xlims,
         r"Figure 4 : Spectrum of $\left(1+0.1\cos\left(t\right)\right)\cos\left(10t\right)$", "4")
t, Y, w = findFFT(-4*pi, 4*pi, 512, f, 3)
Xlims = [-15, 15]
Ylims = []
plot_FFT(t, Y, w, 1e-4, Xlims, r"Figure 5: Spectrum of $\sin ^{3}(t)$", "5")

t, Y, w = findFFT(-4*pi, 4*pi, 512, f, 4)
Xlims = [-15, 15]
plot_FFT(t, Y, w, 1e-4, Xlims, r"Figure 6: Spectrum of $\cos^{3}(t)$", "6")
t, Y, w = findFFT(-4*pi, 4*pi, 512, f, 5)
Xlims = [-40, 40]
plot_FFT(t, Y, w, 1e-3, Xlims, r"Figure 7: Spectrum of $\cos(20t+5\cos(t))$", "7")
# initial window_size and sampling rate defined
window_size = 2*pi
sampling_rate = 128
# tolerance for error
tol = 1e-15

# normalisation factor derived
norm_factor = (window_size)/(2*pi*(sampling_rate))


'''
For loop to minimize the error by increasing 
both window_size and sampling rate as we made assumption that
when Window_size is large the sinc(w) acts like impulse, so we
increase window_size, similarly sampling rate increased to 
overcome aliasing problems
'''

for i in range(1, 10):

    t, Y, w = findFFT(-window_size/2, window_size/2,
                      sampling_rate, f, 6, norm_factor)

    # actual Y
    actual_Y = (1/sqrt(2*pi))*exp(-pow(w, 2)/2)
    error = (np.mean(np.abs(np.abs(Y)-actual_Y)))
    print("Absolute error at Iteration - %g is : %g" % ((i, error)))

    if(error < tol):
        print("\nAccuracy of the DFT is: %g and Iterations took: %g" %
              ((error, i)))
        print("Best Window_size: %g , Sampling_rate: %g" %
              ((window_size, sampling_rate)))
        break
    else:
        window_size = window_size*2
        sampling_rate = (sampling_rate)*2
        norm_factor = (window_size)/(2*pi*(sampling_rate))


Xlims = [-10, 10]
plot_FFT(t, Y, w, 1e-2, Xlims,
         r"Figure 8: Spectrum of $e^{-\frac{t^{2}}{ 2}}$", "8")

# Plotting actual DFT of Gaussian
plot(w, abs(actual_Y),
     label=r"$\frac{1}{\sqrt{2}\pi} e^{\frac{\ - \omega ^{2}}{2}}$")
title("Exact Fourier Transform of Gaussian")
xlim([-10, 10])
ylabel(r"$Y(\omega) \to$")
xlabel(r"$\omega \to$")
grid()
legend()
show()
