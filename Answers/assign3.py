# Assignment 3 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 6/2/2019

# Importing libraries

from pylab import *
import scipy.special as sp
from pandas import DataFrame
Ao = 1.05
Bo = -0.105

# Function to generate the true data


def g(t, A=1.05, B=-0.105):
    return(A*sp.jv(2, t)+B*t)


# Main function
if __name__ == "__main__":
    # Loading noisy data
    try:
        data = loadtxt("fitting.dat")
    except Exception:
        print("fitting.dat not found!")
        exit()
    # Standard Deviation of Noise
    lspace = logspace(-1, -3, 9)
    # Plotting noisy data
    for i in range(1, 10):
        plot(data[:, 0], data[:, i], label=r'$\sigma_'+str(i)+'$ = ' +
             str(around(lspace[i-1], 4)))
        legend()
    # Plotting true data
    plot(data[:, 0], g(data[:, 0]), '#000000', label="True Value")
    legend()
    grid()
    xlabel("t")
    ylabel("f(t)+noise")
    title("Error vs True Value")
    show()
    # Plotting errorbar
    errorbar(data[::5, 0], data[::5, 1], lspace[0], fmt="ro",
             label="Errorbar")
    legend()
    grid()
    plot(data[:, 0], g(data[:, 0]), '#000000', label="f(t)")
    legend()
    # Annotation
    annotate("Noisy Data",
             (data[5, 0], data[5, 1]), xytext=(40, -40), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    annotate("True Data",
             (data[3, 0], g(data[3, 0])-0.01), xytext=(-20, 35), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    annotate("Noise in data(Red line)",
             (data[5, 0], data[5, 1]-0.06), xytext=(-20, -60), textcoords="offset points", arrowprops={"arrowstyle": "->"})
    xlabel("t")
    title(r"Data points for $\sigma$ = 0.10 along with exact function")
    show()
    # Matrices for holding variable data over t (J_2(t) and t)
    M = c_[sp.jv(2, data[:, 0]), data[:, 0]]
    Corr = c_[[Ao, Bo]]
    # True Data
    p = dot(M, Corr)
    s = 0
    for i in range(size(data[:, 0])):
        s += p[i]**2-g(data[i, 0])**2
    Acheck = linspace(0, 2, 21)
    Bcheck = linspace(-0.2, 0, 21)
    e = zeros((len(Acheck), len(Bcheck)))
    # Error calculation to plot contours
    for i in range(len(Acheck)):
        for j in range(len(Bcheck)):
            e[i, j] = mean(
                square(g(data[:, 0], Acheck[i], Bcheck[j])-g(data[:, 0])))
    # Plotting contours
    cs = contour(Acheck, Bcheck, e, levels=20)
    xlabel("A")
    ylabel("B")
    title(r"Contours of $\epsilon_{ij}$")
    clabel(cs, cs.levels[:5], inline=1, fontsize=10)
    plot([1.05], [-0.105], 'ro')
    grid()
    annotate("Exact Location\n of Minima", (1.05, -0.105),
             xytext=(-50, -30), textcoords="offset points")
    show()
    # Solving for A,B using lstsp
    AB, *rest = lstsq(M, data[:, 1:10], rcond=None)
    Aerr = array([square(AB[0, i]-Ao)
                  for i in range(9)])
    Berr = array([square(AB[1, i]-Bo)
                  for i in range(9)])
    # Plotting the error vs stddev
    plot(logspace(-1, -3, 9), Aerr, '+--',
         linewidth=0.4, label='Aerr', dashes=(8, 10))

    plot(logspace(-1, -3, 9), Berr, '+--',
         linewidth=0.4, label='Berr', dashes=(8, 10))
    xlabel("Noise Standard Deviation")
    title("Variation of error with noise")
    ylabel("MSerror")
    legend()
    grid()
    show()
    # Plotting the log log of error vs stddev
    loglog(logspace(-1, -3, 9), Aerr, 'ro', label="Aerr")
    loglog(logspace(-1, -3, 9), Berr, 'bo', label="Berr")
    legend()
    errorbar(logspace(-1, -3, 9), Aerr, std(Aerr), fmt='ro')
    errorbar(logspace(-1, -3, 9), Berr, std(Berr), fmt='bo')
    title("Variation of error with noise (LogLog plot)")
    xlabel(r'$\sigma_n$')
    ylabel("MSerror")
    grid()
    show()
