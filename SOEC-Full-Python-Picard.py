import numpy as np
from scipy import sparse

# ------------------------------------------------------------------------------
# Parameters & Constants for Phase-Field Model
# ------------------------------------------------------------------------------

M = 1.0 #mobility constant, m^5/(J*s)
kappa = 4.0 #gradient energy constant, J/m
xmax = 50.0 #physical domain size, m
nx = 100 #number of spatial discretization points
dx = xmax/(nx-1) #spatial discretization step, m
x = np.linspace(-xmax/2, xmax/2, nx) #spatial grid
nsteps = 1000 #number of time steps
trans = 0 #number of transition steps between two phases

# ------------------------------------------------------------------------------
# Parameters & Constants for Diffusion 
# ------------------------------------------------------------------------------

from scipy.constants import elementary_charge as e, Avogadro as NA, k as kB

epsilon = 1.0
i_a = 0 #-8e6           # Applied current density [A/m²]
sigma = 1.0          #conductivity (for the current density to electric field proportionality)
k = 1.0 #kB

# ------------------------------------------------------------------------------
# Reference values from literature
# ------------------------------------------------------------------------------
T = 1.0 #1273.0               # K, temperature
c_ref = 1660               # mol/m^3, reference concentration
rho_ref = 1.6e6            # C/m^3, reference charge density
varphi_ref = k * T / e     # V, reference potential
interface_width = 4.0      # nm, width of the phase-field interface (0 < C < 1)
lambda_D = 1.0             # nm, Debye length

#intialize the composition field

comp = np.zeros(nx) #composition field

comp[0:int(nx/2 - trans/2)] = 0.0 #left phase

comp[int(nx/2 + trans/2):nx] = 1.0 #right phase

for i in range (trans):
    comp[int(nx/2 - trans/2 + i)] = 1.0/trans * i #linear transition

print(comp)

# make a dictionary with the dict () constructor for all species

charges = {
    "vac" : 2.0,
    "elec" : -1.0,
    "yzr" : -1.0,
}

# concentrations -> [mol/m³]

vac = {
    "mu_YSZ" : 0.12,
    "mu_anode" : 0.2,
    "diffusivity" : 1.0e-8,
    "bulk_anode_conc" : 83.0,
    "bulk_YSZ_conc" : 830.0,
    "rate_constant" : abs(i_a / e * NA * charges["vac"]),
}

elec = {
    "mu_YSZ" : 0.0,
    "mu_anode" : 0.0,
    "diffusivity" : 2.0e-4,
    "bulk_YSZ_conc" : 0.0,
    "bulk_anode_conc" : charges["vac"] * vac["bulk_anode_conc"],
    "rate_constant" : abs(i_a / e * NA * charges["elec"]),
}

yzr = {
    "mu_YSZ" : 0.0,
    "mu_anode" : 0.0,
    "diffusivity" : 5.0e-20,
    "bulk_YSZ_conc" : 1660.0,
    "bulk_anode_conc" : 0.0,
    "rate_constant" : 0.0,
}

conc_boundaries = {
    "vac" : dict(YSZ = ( i_a / e * NA * charges["vac"] ) / vac["diffusivity"], anode = 0.0),
    "elec" : dict(YSZ = 0.0, anode = (i_a / e * NA * charges["elec"]) / vac["diffusivity"]),
    "yzr" : dict(YSZ = 0.0, anode = 0.0),
}

'''
Crank-Nicolson method to solve 1D reaction-diffusion equation:
        u_t = D * u_xx
    
with Neumann boundary conditions 
at x=0: u_x = 0
at x=L: u_x = 0 

with L = 50.0 and initial condition:
u(x,0) = comp
'''

def CrankNicolson(n, comp_init, T = 1.0, L = xmax, M = 1.0, kappa = 4.0):
    #Parameters needed to solve the equation within the explicit method
    # M = GRID POINTS on space interval
    N = (n**2) #GRID POINTS on time interval

    # ---- Length of the wire in x direction ---- 
    x0, xL = -L/2, L/2

    # ----- Spatial discretization step -----
    dx = (xL - x0)/(n-1)

    # ---- Final time ---- 
    t0, tF = 0, T 

    # ----- Time step -----
    dt = (tF - t0)/(N-1)

    # ----- Initialize matrix for diffusion equation -----

    lambda_diff = dt*M/(2.0*dx**2)

    a0 = 1 + 2*lambda_diff
    c0 = 1 - 2*lambda_diff

    xspan = np.linspace(x0, xL, n)
    tspan = np.linspace(t0, tF, N)

    maindiag_a0 = a0*np.ones((1,n))
    offdiag_a0 = (-lambda_diff)*np.ones((1, n-1))

    maindiag_c0 = c0*np.ones((1,n))
    offdiag_c0 = lambda_diff*np.ones((1, n-1))

    #Left-hand side tri-diagonal matrix
    a = maindiag_a0.shape[1]
    diagonalsA = [maindiag_a0, offdiag_a0, offdiag_a0]
    A = sparse.diags(diagonalsA, [0,-1,1], shape=(a,a)).toarray()
    
    A[0,1] = (-2)*lambda_diff
    A[n-1,n-2] = (-2)*lambda_diff

    #Right-hand side tri-diagonal matrix
    c = maindiag_c0.shape[1]
    diagonalsC = [maindiag_c0, offdiag_c0, offdiag_c0]
    
    Arhs = sparse.diags(diagonalsC, [0,-1,1], shape=(c,c)).toarray()
    Arhs[0,1] = 2*lambda_diff
    Arhs[n-1,n-2] = 2*lambda_diff

    # ----- Initialize matrix for the laplacian operator ---- (NEW FOR CAHN-HILLIARD)
    # this is to compute the second derivative of c in the chemical potential
    lambda_lap = 1.0 / dx**2
    maindiag_lap = -2 * lambda_lap * np.ones((1, n))
    offdiag_lap = lambda_lap * np.ones((1, n-1))
    diagonalsLap = [maindiag_lap, offdiag_lap, offdiag_lap]
    laplacian = sparse.diags(diagonalsLap, [0, -1, 1], shape=(n, n)).toarray()

    #modify the first and last rows for Neumann boundary conditions
    laplacian[0, 0] = - lambda_lap
    laplacian[0, 1] = lambda_lap
    laplacian[n-1, n-1] = - lambda_lap
    laplacian[n-1, n-2] = lambda_lap
    
    def df_dC(C):
        """Derivative of the free energy density."""
        return 4 * C * (1 - C) * (1 - 2 * C)

    # ----- Initializes matrix U -----
    U = np.zeros((n, N))

    #----- Initial condition -----
    U[:,0] = comp_init.copy()

    #----- Neumann boundary conditions -----
    #Add one line above and one line below using finite differences 
    leftBC = np.arange(1, N+1)
    f = 0 * leftBC

    rightBC = np.arange(1, N+1)
    g = 0 * rightBC

    #this is where the most changes have been made:

    for k in range(1, N):
        #first, get the current solution:
        u_current = U[0:n, k-1]

        # Intialize for Picard iteration:
        u_next = u_current.copy()

        # Picard iteration for nonlinearity:
        for iter in range(5):
            df = df_dC(u_next)
            lap_u = laplacian @ u_next  # Second derivative - this is the explicit part (?)
            mu = df - kappa * lap_u  # Chemical potential

            #boundary conditions:
            ins = np.zeros((n-2,1)).ravel()
            b1 = np.asarray([4*lambda_diff*dx*f[k], 4*lambda_diff*dx*g[k]])
            b1 = np.insert(b1, 1, ins)

            #explicit part from the previous time step:
            b2 = Arhs @ u_current  # Right-hand side

            #add the chemical potential term:
            b3 = dt * M * laplacian @ mu  # Explicit term

            b = b1 + b2 + b3  # Right hand side
            # Solve the linear system:
            u_next = np.linalg.solve(A, b)

        U[:, k] = u_next  # Update solution

    return (U, tspan, xspan)
#-------------------------------------------------------------------
# Note: The above code is a complete implementation of the Cahn-Hilliard phase-field model using both Crank-Nicolson and Tridiagonal methods.

#initialize the concentrations of each species and electrostatic potential of the system

# ------------------------------------------------------------------------------
# Vacancy concentrations [mol/m³]
# ------------------------------------------------------------------------------

c_vac = np.zeros(nx)

c_vac[0:int(nx/2)] = vac["bulk_YSZ_conc"]
c_vac[int(nx/2):nx] = vac["bulk_anode_conc"]

c_vac = c_vac / c_ref  # normalize by reference concentration

# ------------------------------------------------------------------------------
# Yttrium-stabilized Zirconium concentrations [mol/m³]
# ------------------------------------------------------------------------------

c_yzr = np.zeros(nx)

c_yzr[0:int(nx/2)] = yzr["bulk_YSZ_conc"]
c_yzr[int(nx/2):nx] = yzr["bulk_anode_conc"]

c_yzr = c_yzr / c_ref  # normalize by reference concentration

# ------------------------------------------------------------------------------
# Electron concentrations [mol/m³]
# ------------------------------------------------------------------------------

c_elec = np.zeros(nx)

c_elec[0:int(nx/2)] = elec["bulk_YSZ_conc"]
c_elec[int(nx/2):nx] = elec["bulk_anode_conc"]

c_elec = c_elec / c_ref  # normalize by reference concentration

# ------------------------------------------------------------------------------
# Initialize the electrostatic potential field
# ------------------------------------------------------------------------------

varphi = np.zeros(nx) #potential field

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# function to calculate the charge density, rho

# function to calculate the potential, varphi 

varphi_init = np.zeros(nx)  # Initial potential field
varphi_init[:nx//2] = 0.0  # Left side potential
varphi_init[nx//2:] = 0.0 # Right side potential

def gauss_seidel_1D(varphi_s, rho, i_a, sigma, maxiter = 10000, tol = 1e-4):
    varphi = varphi_s.copy()  # Make a copy of the initial potential
    # Get the length of potential array (1-D)
    N = len(varphi)
    
    # Iterate until convergence or maximum iterations reached
    for it in range(maxiter):
        varphi_old = varphi.copy()

        varphi[0] = varphi[1] + dx * 0.0
        
        varphi[-1] = varphi[-2] + dx * i_a/sigma # Boundary condition at the right end

        # Update the potential using the Gauss-Seidel formula
        for j in range(1, N-1):
            varphi[j] = 0.5 * (varphi[j+1] + varphi[j-1] + rho[j] * dx**2 / epsilon)
        
        # Check for convergence
        if np.max(np.abs(varphi - varphi_old)) < tol:
            print(f"Converged after {it} iterations.")
            return varphi
    
    print("Maximum iterations reached without convergence.")
    return varphi

def laplacian_chem(f, deltax, mu_YSZ, mu_anode):
    f[0] = mu_YSZ
    f[-1] = mu_anode
    lap = np.zeros_like(f)
    for i in range(1, len(f)-1):
        lap[i] = (f[i-1] - 2*f[i] + f[i+1]) / (deltax ** 2)
    return lap

def laplacian_conc(f, deltax, left, right):
    lap = np.zeros_like(f)

    for i in range(1, len(f)-1):

        lap[i] = (f[i-1] - 2*f[i] + f[i+1]) / (deltax ** 2)
    
    # use ghost point to enforce boundary conditions on the left and right sides (again, neumann)
    lap[0] = (f[1] - 2*f[0] + (f[1] - 2 * deltax * left)) / (deltax ** 2)
    lap[-1] = (f[-2] - 2*f[-1] + (f[-2] + 2 * deltax * right)) / (deltax ** 2)
    return lap

def laplacian_elec(f, deltax):
    # Dirichlet boundary condition for the electric potential for electrolyte
    lap = np.zeros_like(f)
    f[0] = 0.0
    for i in range(1, len(f)-1):
        lap[i] = (f[i-1] - 2*f[i] + f[i+1]) / (deltax ** 2)
    lap[-1] = (f[-2] - 2*f[-1] + (f[-2] + 2 * deltax * i_a / sigma)) / (deltax ** 2)

    return lap

# Okay now for the chemical potential

def interpolator(C):
    # Interpolation function for the chemical potential, polynomial
    h = C**3 * ( 6 * C**2 - 15 * C + 10)
    return h
    
# function to calculate the chemical potential, mu
def chemical_potential(C, mu_YSZ, mu_anode):
    mu = mu_YSZ + (mu_anode - mu_YSZ) * interpolator(C)
    return mu

def interface_locator(C):
    # ensures interface reaction only occurs at interface
    lambda_C = C ** 2 * (1 - C) ** 2
    return lambda_C

# maybe because its in reduced units, we need to scale the diffusivity by the reference concentration

def diffusion (varphi, C, c, mu, D, z, R, left, right, mu_YSZ, mu_anode):
    # first term = chemical potential term
    chem_term = ( D * c / ( k * T )) * laplacian_chem(mu, dx, mu_YSZ, mu_anode)
    constants = D * c / ( k * T )

    # second term = concentration gradient term
    conc_term = D * laplacian_conc(c, dx, left, right)

    # third term = electrostatic potential term
    elec_term = ( D * z * e / ( kB * T )) * laplacian_elec(varphi, dx)

    # fourth term = sink term
    # need to first define K (normalization constant)
    integral = np.trapezoid(interface_locator(C), dx = dx) 
    K  = 1.0 / integral
    sink_term = R * K * interface_locator(C)

    #print("sink_term", sink_term)

    c += (chem_term + conc_term + elec_term + sink_term) * dt

    return c

#intialize the composition field

comp = np.zeros(nx) #composition field

comp[0:int(nx/2 - trans/2)] = 0.0 #left phase

comp[int(nx/2 + trans/2):nx] = 1.0 #right phase

for i in range (trans):
    comp[int(nx/2 - trans/2 + i)] = 1.0/trans * i #linear transition

print(comp.shape)

# ------------------------------------------------------------------------------
print(comp.shape)
print("comp", interpolator(comp).shape)
U, tspan, xspan = CrankNicolson(nx, comp, T = 1.0, L = xmax, M = 1.0, kappa = 4.0)
comp = U

#print("U.shape:", U.shape)

for time in range(nsteps):

    #for Poisson's equation (not species-specific)
    rho = - e * NA * ( charges["vac"] * c_vac + charges["elec"] * c_elec + charges["yzr"] * c_yzr )
    rho /= rho_ref  # normalize by reference charge density
    #print("rho",rho)
    
    varphi = gauss_seidel_1D(varphi_init, rho, i_a, sigma)

    #find the chemical potential for every species:
    print(comp.shape)
    mu_vac = chemical_potential(comp, vac["mu_YSZ"], vac["mu_anode"])
    mu_elec = chemical_potential(comp, elec["mu_YSZ"], elec["mu_anode"])
    mu_yzr = chemical_potential(comp, yzr["mu_YSZ"], yzr["mu_anode"])

    print("mu_vac.shape:", mu_vac.shape)

    #update the concentration for every species: 
    c_vac = diffusion(varphi, comp, c_vac, mu_vac, vac["diffusivity"], charges["vac"], vac["rate_constant"], conc_boundaries["vac"]["YSZ"], conc_boundaries["vac"]["anode"], vac["mu_YSZ"], vac["mu_anode"])
    c_elec = diffusion(varphi, comp, c_elec, mu_elec, elec["diffusivity"], charges["elec"], elec["rate_constant"], conc_boundaries["elec"]["YSZ"], conc_boundaries["elec"]["anode"], elec["mu_YSZ"], elec["mu_anode"])
    c_yzr = diffusion(varphi, comp, c_yzr, mu_yzr, yzr["diffusivity"], charges["yzr"], yzr["rate_constant"], conc_boundaries["yzr"]["YSZ"], conc_boundaries["yzr"]["anode"], yzr["mu_YSZ"], yzr["mu_anode"])
