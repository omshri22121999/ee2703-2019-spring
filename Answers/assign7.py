# Assignment 7 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 15/3/2019
# %%

# Importing libraries and setting up plots
from pylab import *
from sympy import *
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
plt.rcParams['font.family'] = "sans"


# %%
'''
function to solve for V(s) by Matrix inversion
This function used for Low pass filter
arguments : R1,R2,C1,C2,G
- parameters of the circuit
Vi - Laplace transform of Input.
'''


def lowpass(R1, R2, C1, C2, G, Vi):
    s = symbols('s')
    A = Matrix([[0, 0, 1, -1/G], [-1/(1+s*R2*C2), 1, 0, 0],
                [0, -G, G, 1], [-1/R1-1/R2-s*C1, 1/R2, 0, s*C1]])
    b = Matrix([0, 0, 0, Vi/R1])
    V = A.inv()*b
    return (A, b, V)


# %%
'''
function to solve for Transfer function H(s)
To convert sympy polynomial to sp.lti polynomial
Arguments : num_coeff
- array of coefficients of denominator polynomial
den_coeff
- array of coefficients of denominator polynomial
Returns
: Hs
- Transfer function in s domain
'''


def sympytolti(n_coeff, d_coeff):
    n_coeff = np.array(n_coeff, dtype=float)
    d_coeff = np.array(d_coeff, dtype=float)
    H_n = poly1d(n_coeff)
    H_d = poly1d(d_coeff)
    Hs = sp.lti(H_n, H_d)
    return Hs

# %%


'''
function to solve for Output voltage for given circuit
Arguments : R1,R2,C1,C2,G
- parameters of the circuit
Vi - Laplace transform of input Voltage
circuitResponse - function defined which is either lpf or Hpf
Returns
: v,Vlti
- v is array of values in jw domain
- Vlti is sp.lti polynomial in s
'''


def solve_circuit(R1, R2, C1, C2, G, Vi, input_freqresponse):
    s = symbols('s')
    A, b, V = input_freqresponse(R1, R2, C1, C2, G, Vi)
    Vo = V[3]

    w = logspace(0, 8, 801)
    ss = 1j*w
    hf = lambdify(s, Vo, "numpy")
    v = hf(ss)

    # Calculating Quality factor for the system
    if(Vi == 1):
        # Vi(s)=1 means input is impulse
        Q = sqrt(1/(pow(den_coeffs[1]/den_coeffs[2],
                        2) / (den_coeffs[0]/den_coeffs[2])))
        print("Quality factor of the system : %g" % (Q))
        return v, Vlti, Q
    else:
        return v, Vlti


# %%
# Declaring params of the circuit1
R1 = 10000
R2 = 10000
C1 = 1e-9
C2 = 1e-9
G = 1.586
# w is x axis of bode plot
s = symbols('s')
w = logspace(0, 8, 801)
Vi_1 = 1  # Laplace transform of impulse
Vi_2 = 1/s  # Laplace transform of u(t)

# Impulse response of the circuit
Vo1, Vs1, Q = solve_circuit(R1, R2, C1, C2, G, Vi_1, lowpass)
# To find Output Voltage in time domain
t1, Vot1 = sp.impulse(Vs1, None, linspace(0, 1e-2, 10000))
# Step response of circuit
Vo2, Vs2 = solve_circuit(R1, R2, C1, C2, G, Vi_2, lowpass)
# To find Output Voltage in time domain
t2, Vot2 = sp.impulse(Vs2, None, linspace(0, 1e-3, 10000))
# %%
# Magnitude response for impulse (in loglog)
loglog(w, abs(Vo1),
       label=r"$|H(j\omega)|$")
title(r"Figure 1a: $|H(j\omega)|$ : Magnitude response of Transfer function")
legend()
xlabel(r"$\omega \to $")
ylabel(r"$ |H(j\omega)| \to $")
grid()
show()
# %%
# Plot of output for step input
step([t2[0], t2[-1]], [0, 1], label=r"$V_{i}(t) = u(t)$")
plt.plot(t2, abs(Vot2), label=r"Response for $V_{i}(t) = u(t)$")
legend()
title(r"Figure 1b: $V_{o}(t)$ : Unit Step response in time domain")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
show()
# %%
# Input sinusoid frequencies in rad/s
w1 = 2000*np.pi
w2 = 2*1e6*np.pi

ts = np.linspace(0, 0.005, 8001)

vi = np.sin(w1*ts)+np.cos(w2*ts)
t, Vout, svec = sp.lsim(Vs1, vi, ts)


# Plotting the output for sinusoid
plt.plot(ts, Vout, label=r"Response for $V_{i}(t) = $ Sinusoid")
legend()
title(r"Figure 2: $V_{o}(t)$ : Output Voltage for sinusoidal input")

xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
plt.ylim([-1.1, 1.1])
grid()
show()
# %%
'''
function to solve for V(s) by Matrix inversion
This function used for High pass filter
arguments : R1,R3,C1,C2,G
- parameters of the circuit
Vi - Laplace transform of Input.
'''


def highpass(R1, R3, C1, C2, G, Vi):
    s = symbols('s')
    A = Matrix([[0, 0, 1, -1/G],
                [(-s)*C2*R3/(1+s*R3*C2), 1, 0, 0],
                [0, -G, G, 1],
                [(-1-(s*R1*C1)-(s*R3*C2)), s*C2*R1, 0, 1]])
    b = Matrix([0, 0, 0, -Vi*s*C1*R1])
    V = A.inv()*b
    return (A, b, V)


# %%
# Params for 2nd circuit
R1b = 10000
R3b = 10000
C1b = 1e-9
C2b = 1e-9
Gb = 1.586
# input frequencies for damped sinusoids
w1 = 2000*np.pi
w2 = 2e6*np.pi
# Decay factor for damped sinusoid
a = 1e5

# Laplace transform of impulse
Vi_1b = 1
# Laplace of unit step
Vi_2b = 1/s

# Solve for step response
Vo1b, Vs1b, Qb = solve_circuit(R1b, R3b, C1b, C2b, Gb, Vi_1b, highpass)
t1b, Vot1b = sp.impulse(Vs1b, None, linspace(0, 1e-2, 10000))

# Solve for impulse response
Vo2b, Vs2b = solve_circuit(R1b, R3b, C1b, C2b, Gb, Vi_2b, highpass)
t2b, Vot2b = sp.impulse(Vs2b, None, linspace(0, 5e-4, 1000001))

# Solving for damped and non-damped sinusoids
ts1 = np.linspace(0, 0.000005, 800001)
ts2 = np.linspace(0, 0.00003, 80001)

V_i3b = np.sin(w1*ts1)+np.cos(w2*ts1)

V_i4b = np.exp(-a*ts2)*(np.sin(w1*ts2)+np.cos(w2*ts2))

t3b, Vot3b, svec = sp.lsim(Vs1b, V_i3b, ts1)

t4b, Vot4b, svec = sp.lsim(Vs1b, V_i4b, ts2)
# %%
# plot of impulse response(Bode)
loglog(w, abs(Vo1b), label=r"$|H(j\omega)|$")
legend()
title(r"Figure 3a: $|H(j\omega)|$ : Magnitude response of Transfer function")
xlabel(r"$\omega \to $")
ylabel(r"$ |H(j\omega)| \to $")
grid()
show()

# %%
# plot of Vo(t) for step input
step([t2b[0], t2b[-1]], [0, 1], label=r"$V_{i}(t) = u(t)$")
plt.plot(t2b, Vot2b, label=r"Unit Step Response for $V_{i}(t) = u(t)$")
legend()
title(r"Figure 3b: $V_{o}(t) $ : Unit step response in time domain")
xlabel(r"$ t (seconds) \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
show()

# %%
# plot of Vo(t) for undamped sinusoidal input
plt.plot(t3b, (Vot3b), label=r"Response for $V_{i}(t) = $ Undamped Sinusoid")
legend()
title(r"Figure 4: $V_{o}(t)$ : Output Voltage for undamped sinusoid input")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
plt.ylim([-1.1, 1.1])
grid()
show()


# %%
# plot of Vo(t) for damped sinusoidal input
plt.plot(t4b, (Vot4b), label=r"Response for $V_{i}(t) = $ Damped Sinusoid")
legend()
title(r"Figure 5: $V_{o}(t)$ : Output Voltage for damped sinusoid input")
xlabel(r"$t \to $")
ylabel(r"$ V_{o}(t) \to $")
grid()
show()


# %%
