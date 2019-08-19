# Assignment 10 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 6/4/2019
# %%


# Importing libraries
import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams['figure.autolayout'] = False
plt.rcParams['figure.figsize'] = 9, 6
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
plt.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}']
# %%
# Reading the frequency coefficients of filter from csv file
fil = pd.read_csv("/home/omshripc/Sem 4/EE2703/Answers/h.csv",
                  header=None)

fil = fil.values
fil = fil[:, 0]
# Reading the frequency response of the filter
w, h = sp.freqz(fil)


# Plotting the magnitude response of the filter
plt.title("Magnitude Response of Filter")
plt.plot(w, np.abs(h))

plt.ylabel(r"$|H(j\omega)| \to$")
plt.xlabel(r"$w \to$")

plt.show()

# Plotting the frequency response of the filter
ph = np.unwrap(np.angle(h))
plt.title("Phase Response of Filter")
plt.plot(w, ph)

plt.ylabel(r"$\angle H(j\omega) \to$")
plt.xlabel(r"$w \to$")
plt.show()
# %%
# Creating the input signal
n = np.linspace(0, 1024, 1025)
n = n[:-1]
x = np.cos(np.pi*0.2*n)+np.cos(0.85*np.pi*n)

plt.title(r"$x[n] = cos(0.2\pi n)+cos(0.85\pi n)$")
plt.stem(n, x)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$x[n] \to$")
plt.show()

plt.title(r"$x[n]$ for n $\in$ [0,100]")
plt.stem(n, x)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$x[n] \to$")
plt.xlim([0, 100])
plt.show()

# %%

# Linear convolution
y_conv = np.convolve(x, fil)

# Plotting linear convolution
plt.title(r"$y[n] = x[n]*h[n]$")
plt.stem(y_conv)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y[n] \to$")
plt.show()

plt.xlim([0, 100])
plt.title(r"$y[n]$ zoomed to start")
plt.stem(y_conv)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y[n] \to$")
plt.show()


plt.xlim([900, 1050])
plt.title(r"$y[n]$ zoomed to end")
plt.stem(y_conv)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y[n] \to$")
plt.show()


# %%

# Circular convolution

fil_rep = np.concatenate([fil, np.zeros(len(x) - len(fil))])
y1 = np.fft.ifft(np.fft.fft(x)*np.fft.fft(fil_rep))

# Plotting circular convolution
plt.title(r"$y'[n] = x[n] \circledast h[n]$")
plt.stem(y1)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y'[n] \to$")
plt.show()

plt.xlim([0, 100])
plt.title(r"$y'[n]$ zoomed to start")
plt.stem(y1)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y'[n] \to$")
plt.show()


plt.xlim([900, 1050])
plt.title(r"$y'[n]$ zoomed to end")
plt.stem(y1)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y'[n] \to$")
plt.show()
# %%
# Doing linear convolution using circular convolution

# Splitting x into different parts
x_split = np.array(np.split(x, 1024/16))
P = len(fil)
L = x_split.shape[1]
N = L+P-1
# Creating space to store the output
y_part = np.zeros((x_split.shape[0], N))

# Padding the filter with zeros
fil_new = np.concatenate([fil, np.zeros(N-len(fil))])


# Finding circular convolution of each part of signal
for i in range(x_split.shape[0]):
    x_new = np.concatenate([x_split[i], np.zeros(N-len(x_split[i]))])
    y_part[i] = np.array(np.fft.ifft(np.fft.fft(x_new)*np.fft.fft(fil_new)))
y = np.zeros(len(x)+len(fil))


# Creating output by summing up all parts
J = len(y_part[0])
K = J-(P-1)
for i in range(y_part.shape[0]):
    y[i*K:i*K+J] += y_part[i]


plt.title(r"$y[n] = x[n]*h[n]$ using circular convolution")
plt.stem(y)
plt.xlabel(r"$n \to$")
plt.ylabel(r"$y'[n] \to$")
plt.show()
# %%

# Analysing the Zandoff-Chu sequence
z_c = pd.read_csv("/home/omshripc/Sem 4/EE2703/Answers/x1.csv",
                  header=None).values[:, 0]

z_c = np.array([complex(z_c[i].replace("i", "j")) for i in range(len(z_c))])


# Plotting magnitude of sequence
plt.title(r"Magnitude of Zadoff-Chu sequence ($z[n]$)")
plt.plot(np.round(np.abs(z_c)), 'bo')

plt.xlabel(r"$n \to$")
plt.ylabel(r"$z[n] \to$")
plt.show()

# %%

# Cyclic shifted Zandoff-Chu sequence of shift = 5
z_c_rot = np.concatenate([z_c[5:], z_c[0:5]])
# Linear convolution of Zandoff-Chu sequence with cyclic shifted version
y_corr = np.correlate(z_c, z_c_rot, "full")
n = np.linspace(-len(z_c), len(z_c), len(y_corr))

# Plotting the linear convolution
plt.title(r"$p[n]$ = Correlation of $z[n]$ with z[n] cyclically shifted by n=5")

plt.stem(n, np.abs(y_corr))
plt.xlabel(r"$n \to$")
plt.ylabel(r"$p[n] \to$")
plt.show()

plt.title(r"$p[n]$ for n $\in$ [3,7]")
plt.stem(n, np.abs(y_corr))
plt.xlim([3, 7])
plt.xlabel(r"$n \to$")
plt.ylabel(r"$p[n] \to$")
plt.show()

# %%
# Circular convolution of Zandoff-Chu sequence with cyclic shifted version
y_cir_conv = np.fft.ifft(np.fft.fft(z_c)*np.fft.fft(np.conj(z_c_rot)))
plt.title(
    r"$q[n]$ = Circular Correlation of $z[n]$ with z[n] cyclically shifted by n=5")

# Plotting the circular convolution
plt.stem(np.abs(y_cir_conv))
plt.xlabel(r"$n \to$")
plt.ylabel(r"$p[n] \to$")
plt.show()

plt.title(r"$q[n]$ for n $\in$ [830,838]")
plt.stem(np.abs(y_cir_conv))
plt.xlim([830, 838])
plt.xlabel(r"$n \to$")
plt.ylabel(r"$p[n] \to$")
plt.show()


# %%
