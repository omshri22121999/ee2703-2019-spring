# Assignment 4 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 23/2/2019

# Importing libraries
from pylab import *
from scipy.integrate import quad

# # Question 1
# * Define Python functions for the two functions $e^{x}$ and $\cos(\cos(x))$ which return a vector (or scalar) value.
# * Plot the functions over the interval [−2$\pi$,$4\pi$).
# * Discuss periodicity of both functions
# * Plot the expected functions from fourier series

# Functions for $e^{x}$ and $\cos(\cos(x))$ is defined


def exp_fn(x):
    return exp(x)


def coscos_fn(x):
    return cos(cos(x))


x = linspace(-2*pi, 4*pi, 400)
# Period of function created using fourier coefficients will be 2pi
period = 2*pi


# Plotting original function vs expected function for exp(x)
semilogy(x, exp_fn(x), 'k', label="Original Function")
# plotting expected function by dividing the x by period and giving remainder as
# input to the function, so that x values repeat after given period.
semilogy(x, exp_fn(x % period), '--',
         label="Expected Function from fourier series")
legend()
title(r"Figure 1 : Plot of $e^{x}$")
xlabel(r"x")
ylabel(r"$e^{x}$")
grid()

show()
# Plotting original function vs expected function for cos(cos((x)))
plot(x, coscos_fn(x), 'b', linewidth=4, label="Original Function")
# plotting expected function by dividing the x by period and giving remainder as
# input to the function, so that x values repeat after given period.
plot(x, coscos_fn(x % period), 'y--',
     label="Expected Function from fourier series")
legend(loc='upper right')
title(r"Figure 2 : Plot of $\cos(\cos(x))$")
xlabel(r"x")
ylabel(r"$\cos(\cos(x))$")
plt.axis([-5, 5, -0.5, 2])
grid()

show()

# # Question 2
# * Obtain the first 51 coefficients i.e $a_{0}, a_{1}, b_{1},....$ for $e^{x}$ and $\cos(\cos(x))$ using scipy quad function
# * And to calculate the function using those coefficients and comparing with original funcitons graphically.

# function to calculate the coefficient integrand


def an_fourier(x, k, f):
    return f(x)*cos(k*x)


def bn_fourier(x, k, f):
    return f(x)*sin(k*x)

# function to find the fourier coefficients taking function 'f' as argument.


def find_coeff(f):

    coeff = []
    coeff.append((quad(f, 0, 2*pi)[0])/(2*pi))
    for i in range(1, 26):
        coeff.append((quad(an_fourier, 0, 2*pi, args=(i, f))[0])/pi)
        coeff.append((quad(bn_fourier, 0, 2*pi, args=(i, f))[0])/pi)

    return coeff


# function to create 'A' matrix for calculating function back from coefficients
# with no_of rows, columns and vector x as arguments
def matrix_create(nrow, ncol, x):
    A = zeros((nrow, ncol))  # allocate space for A
    A[:, 0] = 1  # col 1 is all ones
    for k in range(1, int((ncol+1)/2)):
        A[:, 2*k-1] = cos(k*x)  # cos(kx) column
        A[:, 2*k] = sin(k*x)  # sin(kx) column
    # endfor
    return A


# Function to compute function from coefficients with argument as coefficient vector 'c'
def compute_fn(c):
    A = matrix_create(400, 51, x)
    f_fourier = A.dot(c)
    return f_fourier


# Initialising empty lists to store coefficients for both functions
exp_coeff = []
coscos_coeff = []
exp_coeff1 = []
coscos_coeff1 = []

exp_coeff1 = find_coeff(exp_fn)
coscos_coeff1 = find_coeff(coscos_fn)

# to store absolute value of coefficients
exp_coeff = np.abs(exp_coeff1)
coscos_coeff = np.abs(coscos_coeff1)

# Computing function using fourier coeff
exp_fn_fourier = compute_fn(exp_coeff1)
coscos_fn_fourier = compute_fn(coscos_coeff1)


# Plotting the Function computed using Fourier Coefficients
semilogy(x, exp_fn_fourier, 'ro', label="Function using Fourier Coefficients")
ylim([pow(10, -1), pow(10, 4)])
legend()

grid()
show()


plot(x, coscos_fn_fourier, 'ro', label="Function using Fourier Coefficients")
legend(loc='upper right')

plt.axis([-5, 5, -0.5, 2])
grid()
show()


# # Question3
# * Two different plots for each function using “semilogy” and “loglog” and plot the magnitude of the coefficients vs n
# * And to analyse them and to discuss the observations.
# ## Plots:
# * For each function magnitude of $a_{n}$ and $b_{n}$ coefficients which are computed using integration are plotted in same figure in semilog as well as loglog plot for simpler comparisons.

# By using array indexing methods we separate all odd indexes starting from 1 -> an
# and all even indexes starting from 2 -> bn
semilogy((exp_coeff[1::2]), 'ro', label=r"$a_{n}$ using Integration")
semilogy((exp_coeff[2::2]), 'ko', label=r"$b_{n}$ using Integration")
legend()
title("Figure 3 : Fourier coefficients of $e^{x}$ (semi-log)")
xlabel("n")
ylabel("Magnitude of coeffients")

grid()
show()

# By using array indexing methods we separate all odd indexes starting from 1 -> an
# and all even indexes starting from 2 -> bn
loglog((exp_coeff[1::2]), 'ro', label=r"$a_{n}$ using Integration")
loglog((exp_coeff[2::2]), 'ko', label=r"$b_{n}$ using Integration")
legend(loc='upper right')
title("Figure 4 : Fourier coefficients of $e^{x}$ (Log-Log)")
xlabel("n")

grid()
ylabel("Magnitude of coeffients")
show()


# By using array indexing methods we separate all odd indexes starting from 1 -> an
# and all even indexes starting from 2 -> bn
semilogy((coscos_coeff[1::2]), 'ro', label=r"$a_{n}$ using Integration")
semilogy((coscos_coeff[2::2]), 'ko', label=r"$b_{n}$ using Integration")
legend(loc='upper right')
title("Figure 5 : Fourier coefficients of $\cos(\cos(x))$ (semi-log)")
xlabel("n")

grid()
ylabel("Magnitude of coeffients")
show()

# By using array indexing methods we separate all odd indexes starting from 1 -> an
# and all even indexes starting from 2 -> bn
loglog((coscos_coeff[1::2]), 'ro', label=r"$a_{n}$ using Integration")
loglog((coscos_coeff[2::2]), 'ko', label=r"$b_{n}$ using Integration")
legend(loc='upper right')
title("Figure 6 : Fourier coefficients of $\cos(\cos(x))$  (Log-Log)")
xlabel("n")

grid()
ylabel("Magnitude of coeffients")
show()

# # Question 4  & 5
# * Uses least squares method approach to find the fourier coefficients of $e^{x}$ and $\cos(\cos(x))$
# * Evaluate both the functions at each x values and call it b. Now this is approximated by
#   $a_{0} + \sum\limits_{n=1}^{\infty} {{a_{n}\cos(nx)+b_{n}\sin(nx)}}$
# * such that \begin{equation}
#     a_{0} + \sum\limits_{n=1}^{\infty} {{a_{n}\cos(nx_{i})+b_{n}\sin(nx_{i})}} \approx f(x_{i})
#     \end{equation}
# * To implement this we use matrices to find the coefficients using Least Squares method using inbuilt python function 'lstsq'
#

# Function to calculate coefficients using lstsq and by calling
# function 'matrix_create' which was defined earlier in the code
# to create 'A' matrix with arguments as function 'f' and lower
# and upper limits of input x and no_of points needed


def lstsq_coeff(f, low_lim, upp_lim, no_points):
    x1 = linspace(low_lim, upp_lim, no_points)
    # drop last term to have a proper periodic integral
    x1 = x1[:-1]
    b = []
    b = f(x1)
    A = matrix_create(no_points-1, 51, x1)
    c = []
    c = lstsq(A, b, rcond=None)[0]  # the ’[0]’ is to pull out the
    # best fit vector. lstsq returns a list.
    return c


# Calling function and storing them in respective vectors.
coeff_exp = lstsq_coeff(exp_fn, 0, 2*pi, 401)
coeff_coscos = lstsq_coeff(coscos_fn, 0, 2*pi, 401)

# To plot magnitude of coefficients this is used
c1 = np.abs(coeff_exp)
c2 = np.abs(coeff_coscos)


# Plotting in coefficients got using Lstsq in corresponding figures
# 3,4,5,6 using axes.
semilogy((c1[1::2]), 'go', label=r"$a_{n}$ using Least Squares")
semilogy((c1[2::2]), 'bo', label=r"$b_{n}$ using Least Squares")

grid()
legend(loc='upper right')
show()


loglog((c1[1::2]), 'go', label=r"$a_{n}$ using Least Squares ")
loglog((c1[2::2]), 'bo', label=r"$b_{n}$ using Least Squares")

grid()
legend(loc='lower left')
show()


semilogy((c2[1::2]), 'go', label=r"$a_{n}$ using Least Squares")
semilogy((c2[2::2]), 'bo', label=r"$b_{n}$ using Least Squares")


grid()
legend(loc='upper right')
show()

loglog((c2[1::2]), 'go', label=r"$a_{n}$ using Least Squares ")
loglog((c2[2::2]), 'bo', label=r"$b_{n}$ using Least Squares")

grid()
legend(loc=0)
show()


# # Question 6
# * To compare the answers got by least squares and by the direct integration.
# * And finding deviation between them and find the largest deviation using Vectors

# Function to compare the coefficients got by integration and
# least squares and find largest deviation using Vectorized Technique
# Argument : 'integer f which is either 1 or .
# 1 -> exp(x)    2 -> cos(cos(x))


def coeff_compare(f):
    deviations = []
    max_dev = 0
    if(f == 1):
        deviations = np.abs(exp_coeff1 - coeff_exp)
    elif(f == 2):
        deviations = np.abs(coscos_coeff1 - coeff_coscos)

    max_dev = np.amax(deviations)
    return deviations, max_dev


dev1, maxdev1 = coeff_compare(1)
dev2, maxdev2 = coeff_compare(2)

print("Maximum deviation in exp coefficients : ", maxdev1)
print("Maximum deviation in cos_cos coefficients : ", maxdev2)
# Plotting the deviation vs n
plot(dev1, 'g')
title(r"Figure 7 : Deviation between Coefficients for $e^{x}$")

grid()
xlabel("n")
ylabel("Magnitude of Deviations")
show()


# Plotting the deviation vs n
plot(dev2, 'g')
title(r"Figure 8 : Deviation between coefficients for $\cos(\cos(x))$")

grid()
xlabel("n")
ylabel("Magnitude of Deviations")
show()

# # Question 7
#
# * Computing  Ac i.e multiplying Matrix A and Vector C from the estimated values of Coeffient Vector C by Least Squares Method.
# * To Plot them (with green circles) in Figures 1 and 2 respectively for the two functions.


# Define vector x1 from 0 to 2pi
x1 = linspace(0, 2*pi, 400)


# Function to reconstruct the signalfrom coefficients
# computed using Least Squares.
# Takes coefficient vector : 'c' as argument
# returns vector values of function at each x
def fn_create_lstsq(c):
    f_lstsq = []
    A = matrix_create(400, 51, x1)
    f_lstsq = A.dot(c)
    return f_lstsq


exp_fn_lstsq = fn_create_lstsq(coeff_exp)
coscos_fn_lstsq = fn_create_lstsq(coeff_coscos)

# Plotting in Figure1 to compare the original function
# and Reconstructed one using Least Squares method
semilogy(x1, exp_fn_lstsq, 'go',
         label="Inverse Fourier Transform From Least Squares")
legend()

grid()
ylim([pow(10, -2), pow(10, 3)])
xlim([0, 2*pi])
show()

plot(x1, coscos_fn_lstsq, 'go', markersize=4,
     label="Inverse Fourier Transform From Least Squares")
ylim([0.5, 1.3])
xlim([0, 2*pi])
grid()
legend()
show()
