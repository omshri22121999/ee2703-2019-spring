# Assignment 5 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 1/3/2019

# Importing libraries
from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3


# ## Question 1
# ### Part A
# * Define the Parameters, I took $N_x = 50$ and $N_y = 50$ and No of iterations : 6000
# * These values are taken to discuss about Stopping condition,etc
# * To allocate the potential array $\phi = 0$ .Note that the array should have $N_y$ rows and $N_x$ columns.
# * To find the indices which lie inside the circle of radius 0.35 using meshgrid() by equation :
#
# \begin{equation}
# X ∗ X +Y ∗Y ≤ 0.35^2
# \end{equation}
# * Then assign 1 V to those indices.
# * To plot a contour plot of potential $\phi$ and to mark V=1 region in red


Nx = 50  # size along x
Ny = 50  # size along y
radius = 0.35         # radius of central lead
Niter = 6000          # number of iterations to perform


phi = np.zeros((Ny, Nx))          # initialise potential matrix with zeroes
y = linspace(-0.5, 0.5, Ny)  # initialise y range
x = linspace(-0.5, 0.5, Nx)  # initialise x range
'''
Here is '-y' is used inplace of y in meshgrid because we need to get (-0.5,0.5)
at the top left corner of the plate, with center being (0,0)
'''
X, Y = meshgrid(x, -y)            # X,Y coordinates
# indices which lie inside circle of r = 0.35
ii = where(square(X) + square(Y) <= pow(radius, 2))
phi[ii] = 1.0  # assigning V=1 for the circular region


# Plotting contour of potential
fig1 = figure()
ax1 = fig1.add_subplot(111)
plt1 = ax1.contourf(phi, cmap=cm.jet, clabel="$\phi$")
title("Figure 1 : Contour Plot of $\phi$")
ax = gca()
fig1.colorbar(plt1, ax=ax, orientation='vertical')
xlabel("$x$")
ylabel("$y$")
savefig("Figure1.jpg")
show()


# function to create Matrix for finding the Best fit using lstsq
# with no_of rows, columns by default  and vector x as arguments
def createAmatrix(nrow, x):
    A = zeros((nrow, 2))  # allocate space for A
    A[:, 0] = 1
    A[:, 1] = x
    return A


# function to find best fit errors using lstsq
def fitForError(errors, x):
    A = createAmatrix(len(errors), x)
    return A, lstsq(A, log(errors), rcond=None)[0]


# Function to compute function back from Matrix and Coefficients A and B
def computeErrorFit(M, c):
    return exp(M.dot(c))


errors = zeros(Niter)  # initialise error array to zeros
iterations = []  # array from 0 to Niter used for findind lstsq

for k in range(Niter):
    # copy the old phi
    oldphi = phi.copy()

    # Updating the potential
    phi[1:-1, 1:-1] = 0.25 * \
        (phi[1:-1, 0:-2]+phi[1:-1, 2:]+phi[0:-2, 1:-1]+phi[2:, 1:-1])

    # applying boundary conditions
    phi[1:-1, 0] = phi[1:-1, 1]  # Left edge
    phi[1:-1, -1] = phi[1:-1, -2]  # right edge
    phi[0, :] = phi[1, :]  # Top edge
    # Bottom edge is grounded so no boundary conditions

    # Assign 1 V to electrode region
    phi[ii] = 1.0

    # Appending errors for each iterations
    errors[k] = (abs(phi-oldphi)).max()
    iterations.append(k)
# end


fig2 = figure()
ax2 = fig2.add_subplot(111)

# ax2.semilogy(iterations,error_fit1,'r',markersize = 8,label="Fit1")
ax2.semilogy(iterations, errors, 'g', markersize=8, label="Original Error")

ax2.legend()
title(r"Figure 2a : Error Vs No of iterations (Semilog)")
xlabel("$Niter$")
ylabel("Error")
grid()
savefig("Figure2a.jpg")
show()


fig2b = figure()
ax2b = fig2b.add_subplot(111)

# ax2b.loglog(iterations,error_fit1,'r',markersize = 8,label="Fit1")
ax2b.loglog(iterations, errors, 'g', markersize=8, label="Original Error")

ax2b.legend()
title(r"Figure 2b : Error Vs No of iterations (Loglog)")
xlabel("$Niter$")
ylabel("Error")
grid()
savefig("Figure2b.jpg")
show()


# to find the coefficients of fit1 and fit2
# M1 and M2 are matrices and c1 and c2 are coefficients

M1, c1 = fitForError(errors, iterations)  # fit1
M2, c2 = fitForError(errors[500:], iterations[500:])  # fit2

print("Fit1 : A = %g , B = %g" % ((exp(c1[0]), c1[1])))
print("Fit2 : A = %g , B = %g" % ((exp(c2[0]), c2[1])))

print("The time Constant (1/B) all iterations considered: %g" % (abs(1/c1[1])))
print("The time Constant (1/B) for higher iterations (from 500) : %g" %
      (abs(1/c2[1])))


# Calculating the fit using Matrix M and Coefficents C obained
error_fit1 = computeErrorFit(M1, c1)  # fit1
M2new = createAmatrix(len(errors), iterations)

# Error calculated for all iterations using coefficients found using lstsq
error_fit2 = computeErrorFit(M2new, c2)  # fit2 calculated


# Plotting the estimated error_fits using lstsq
fig3 = figure()
ax3 = fig3.add_subplot(111)

# plotted for every 200 points for fit1 and fit2
ax3.semilogy(iterations[0::200], error_fit1[0::200],
             'ro', markersize=8, label="Fit1")
ax3.semilogy(iterations[0::200], error_fit2[0::200],
             'bo', markersize=6, label="Fit2")
ax3.semilogy(iterations, errors, 'k', markersize=6, label="Actual Error")

ax3.legend()
title(r"Figure 3 : Error Vs No of iterations (Semilog)")
xlabel("$Niter$")
ylabel("Error")
grid()
savefig("Figure3.jpg")
show()


def cumerror(error, N, A, B):
    return -(A/B)*exp(B*(N+0.5))


def findStopCondn(errors, Niter, error_tol):
    cum_error = []
    for n in range(1, Niter):
        cum_error.append(cumerror(errors[n], n, exp(c1[0]), c1[1]))
        if(cum_error[n-1] <= error_tol):
            print("last per-iteration change in the error is %g"
                  % (cum_error[-1]-cum_error[-2]))
            return cum_error[n-1], n

    print("last per-iteration change in the error is %g"
          % (np.abs(cum_error[-1]-cum_error[-2])))
    return cum_error[-1], Niter


error_tol = pow(10, -8)
cum_error, Nstop = findStopCondn(errors, Niter, error_tol)
print("Stopping Condition N: %g and Error is %g" % (Nstop, cum_error))


fig5 = figure()  # open a new figure
ax5 = p3.Axes3D(fig5)  # Axes3D is the means to do a surface plot
title("Figure 4: 3-D surface plot of the potential $\phi$")
surf = ax5.plot_surface(X, Y, phi, rstride=1, cstride=1, cmap=cm.jet)
ax5.set_xlabel('$x$')
ax5.set_ylabel('$y$')
ax5.set_zlabel('$z$')
cax = fig5.add_axes([1, 0, 0.1, 1])
savefig("Figure5.jpg")
fig5.colorbar(surf, cax=cax, orientation='vertical')
show()


# #### Contour Plot of the Potential:


fig6 = figure()
ax6 = fig6.add_subplot(111)
plt6 = ax6.contourf(X, Y, phi, cmap=cm.jet)
title("Figure 6 : Contour plot of Updated potential $\phi$")
ax = gca()
fig1.colorbar(plt6, ax=ax, orientation='vertical')
xlabel("$x$")
ylabel("$y$")
grid()
savefig("Figure6.jpg")
show()


Jx = zeros((Ny, Nx))
Jy = zeros((Ny, Nx))

Jx[1:-1, 1:-1] = 0.5*(phi[1:-1, 0:-2] - phi[1:-1, 2:])
Jy[1:-1, 1:-1] = 0.5*(phi[2:, 1:-1] - phi[0:-2, 1:-1])


# #### To Plot the current density using quiver, and mark the electrode via red dots :


fig7 = figure()
ax7 = fig7.add_subplot(111)

ax7.scatter(x[ii[0]], y[ii[1]], color='r', s=12, label="V = 1 region")

ax7.quiver(X, Y, Jx, Jy)
ax7.set_xlabel('$x$')
ax7.set_ylabel('$y$')

ax7.legend()
title("The Vector plot of the current flow")
savefig("Figure7.jpg")
show()
