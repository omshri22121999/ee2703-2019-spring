# Assignment 6 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 9/3/2019

from pylab import *
import scipy.signal as sp


plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 12, 9
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['font.size'] = 16
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['legend.numpoints'] = 2
plt.rcParams['legend.loc'] = 'best'
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.serif'] = "cm"

'''
Function to solve for x(t), by using the sp.impulse function

decay-Decay factor in (s)^-1 of the oscillator
w - Frequency of oscillator in rad/sec

Outputs - Transfer function, x(t) and t for which x(t) is defined
'''


def laplaceSolver(decay, w):
    Xn = poly1d([1, decay])
    Xd = polymul([1, 0, pow(1.5, 2)], [
                 1, 2*decay, (pow(w, 2)+pow(decay, 2))])
    # Computes the impulse response of the transfer function
    Xs = sp.lti(Xn, Xd)
    t, x = sp.impulse(Xs, None, linspace(0, 100, 10000))
    return Xs, t, x


# solving for two cases with decay of 0.5 and 0.05
X, t1, x1 = laplaceSolver(0.5, 1.5)
X, t2, x2 = laplaceSolver(0.05, 1.5)

# plot of x(t) with decay of 0.5
plot(t1, x1, label="decay = 0.5")
legend()
title(r"Figure 1a: $x(t)$ of spring system")
ylim((-1, 1))
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
grid()
show()

# plot of x(t) with decay of 0.05
plot(t2, x2, label="decay = 0.05")
legend()
ylim((-8, 8))
title(r"Figure 1b: $x(t)$ of spring system")
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
grid()
show()

'''
Function to return the function for given decay and frequency

t - Time over which the function must be returned
decay-Decay factor in (s)^-1 of the oscillator
w - Frequency of oscillator in rad/sec

Outputs - x(t)
'''


def cosf(t, w, decay):
    return cos(w*t)*exp(-decay*t)


# Plot of x(t) with different input frequencies
for w in arange(1.4, 1.6, 0.05):
    decay = 0.05
    H, a, b, = laplaceSolver(decay, 1.5)
    t = linspace(0, 200, 20000)
    t, y, svec = sp.lsim(H, cosf(t, w, decay), t)
    plot(t, y, label="$w$ = %g rad/s" % (w))
    legend()
    plot()
xlabel(r"$t \to $")
ylabel(r"$x(t) \to $")
ylim((-80, 80))
title(r"Figure 2: $x(t)$ of spring system with varying frequencies")
grid()
show()

'''
function to solve for Transfer function H(s)
Arguments : n_coeff   - array of coefficients of denominator polynomial
            d_coeff   - array of coefficients of denominator polynomial
Returns   : t,h         - time and response of the system
'''


def coupledSysSolver(n_coeff, d_coeff):
    H_n = poly1d(n_coeff)
    H_d = poly1d(d_coeff)

    Hs = sp.lti(H_n, H_d)
    t, h = sp.impulse(Hs, None, linspace(0, 100, 1000))
    return t, h


# find x and y using above function
t1, x = coupledSysSolver([1, 0, 2], [1, 0, 3, 0])
t2, y = coupledSysSolver([2], [1, 0, 3, 0])

# plot x(t) and y(t)
plot(t1, x, 'b', label="$x(t)$")
plot(t2, y, 'r', label="$y(t)$")
legend()
title(r"Figure 3: Time evolution of $x(t)$ and $y(t)$ for $0 \leq t \leq 100$. of Coupled spring system ")
xlabel(r"$t \to $")
ylabel(r"$x(t),y(t) \to $")
ylim((-0.5, 2))
grid()
show()

'''
function to solve given RLC network for any R,L,C values
Returns   : w,mag,phi,Hs
'''


def RLCnetwork(R, C, L):
    Hnum = poly1d([1])
    Hden = poly1d([L*C, R*C, 1])
    # Computes the impulse response of the transfer function
    Hs = sp.lti(Hnum, Hden)
    # Calculates magnitude and phase response
    w, mag, phi = Hs.bode()
    return w, mag, phi, Hs


# Finds magnitude and phase response of Transfer function
R = 100
L = 1e-6
C = 1e-6

w, mag, phi, H = RLCnetwork(R, L, C)

# plot Magnitude Response
semilogx(w, mag, 'b', label="$Magnitude Response$")
legend()
title(r"Figure 4: Magnitude Response of $H(jw)$ of Series RLC network")
xlabel(r"$ \log w \to $")
ylabel(r"$ 20\log|H(jw)|  \to $")
grid()
show()

# Plot of phase response
semilogx(w, phi, 'r', label="$Phase Response$")
legend()
title(r"Figure 5: Phase response of the $H(jw)$ of Series RLC network")
xlabel(r"$ \log w \to $")
ylabel(r"$ \angle H(j\omega)$ $\to $")
grid()
show()


# Expected output of a ideal LPF
t = linspace(0, 90*pow(10, -3), pow(10, 6))
vi = cos(t*pow(10, 3))-cos(t*pow(10, 6))

# finds Vo(t) using lsim
t, vo, svec = sp.lsim(H, vi, t)
vo_ideal = cos(1e3*t)

# plot of Vo(t) for large time i.e at steady state
# Long term response
plot(t, vo, 'r', label="Output voltage $v_0(t)$ for large time")
legend()
ylim(-2, 2)
title(r"Figure 6a: Output voltage $v_0(t)$  of series RLC network for given $v_i(t)$ at Steady State")
xlabel(r"$ t \to $")
ylabel(r"$ y(t) \to $")
grid()
show()

# plot of Vo(t) in a zoomed manner compared with ideal output
plot(t, vo, 'r', label="Output voltage $v_0(t)$ - zoomed in ")
plot(t, vo_ideal, 'g', label="Ideal Low Pass filter Output with cutoff at $10^4$")
xlim(0.0505, 0.051)
ylim(0.75, 1.1)
legend()
title(r"Figure 6b: Output voltage $v_0(t)$  Vs Ideal Low pass filter Output")
xlabel(r"$ t \to $")
ylabel(r"$ y(t) \to $")
grid()
show()

# Plot of Vo(t) for 0<t<30usec
plot(t, vo, 'r', label="Output voltage $v_0(t)$ : $0<t<30\mu sec$")
legend()
title(r"Figure 7: Output voltage $v_0(t)$ for $0<t<30\mu sec$")
xlim(0, 3e-5)
ylim(-1e-5, 0.3)
xlabel(r"$ t \to $")
ylabel(r"$ v_0(t) \to $")
grid()
show()
