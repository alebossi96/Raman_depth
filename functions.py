import numpy as np
from numpy import sqrt, exp, pi
from scipy.integrate import quad, dblquad, trapz

def A_parameter(ne, ni):
    """
    Computes parameter A; equation (27)
    Args:
    ne: external medium
    ni: diffusing medium
    Parameter A is computed using the solution presented in:
    Contini D, Martelli F, Zaccanti G.
    Photon migration through a turbid slab described by a model
    based diffusion approximation. I. Theory
    Applied Optics Vol 36, No 19, 1997, pp 4587-4599
    """
    # page 4590
    n = ni / ne
    if n == 1:
        # page 4591
        A = 1
    elif n > 1:
        # equations (30)
        t1 = 4 * (-1 - n**2 + 6*n**3 - 10*n**4 - 3*n**5 + 2*n**6 + 6*n**7 - 3*n**8 - (6*n**2 + 9*n**6) * np.sqrt(n**2 - 1)) / (3 * n * (n**2 - 1)**2 * (n**2 + 1)**3)
        t2 = (-8 + 28*n**2 + 35*n**3 - 140*n**4 + 98*n**5 - 13*n**7 + 13*n * (n**2 - 1)**3 * np.sqrt(1 - (1 / n**2))) / (105 * n**3 * (n**2 - 1)**2)
        t3 = 2 * n**3 * (3 + 2 * n**4) * np.log( ((n - np.sqrt(1 + n**2)) * (2 + n**2 + 2 * np.sqrt(1 + n**2)) * (n**2 + np.sqrt(n**4 - 1))) / (n**2 * (n + np.sqrt(1 + n**2)) * (-n**2 + np.sqrt(n**4 - 1))) ) / ((n**2 - 1)**2 * (n**2 + 1)**(7/2))
        t4 = ((1 + 6 * n**4 + n**8) * np.log((-1 + n) / (1 + n)) + 4 * (n**2 + n**6) * np.log((n**2 * (1 + n)) / (n - 1))) / ((n**2 - 1)**2 * (n**2 + 1)**3)
        # equation (29)
        B = 1 + (3 / 2) * (2 * (1 - 1 / n**2)**(3/2) / 3 + t1 + t2 + ((1 + 6 * n**4 + n**8) * (1 - (n**2 - 1)**(3/2) / n**3)) / (3 * (n**4 - 1)**2) + t3)
        C = 1 - (2 + 2 * n - 3 * n**2 + 7 * n**3 - 15 * n**4 - 19 * n**5 - 7 * n**6 + 3 * n**7 + 3 * n**8 + 3 * n**9) / (3 * n**2 * (n - 1) * (n + 1)**2 * (n**2 + 1)**2) - t4
        A = B / C
    else:
        # equations (28)
        r1 = (-4 + n - 4 * n**2 + 25 * n**3 - 40 * n**4 - 6 * n**5 + 8 * n**6 + 30 * n**7 - 12 * n**8 + n**9 + n**11) / (3 * n * (n**2 - 1)**2 * (n**2 + 1)**3)
        r2 = (2 * n**3 * (3 + 2 * n**4)) / ((n**2 - 1)**2 * (n**2 + 1)**(7 / 2)) * np.log((n**2 * (n - np.sqrt(1 + n**2))) * (2 + n**2 + 2 * np.sqrt(1 + n**2)) / ((n + np.sqrt(1 + n**2)) * (-2 + n**4 - 2 * np.sqrt(1 - n**4))))
        r3 = (4 * np.sqrt(1 - n**2) * (1 + 12 * n**4 + n**8)) / (3 * n * (n**2 - 1)**2 * (n**2 + 1)**3)
        r4 = ((1 + 6 * n**4 + n**8) * np.log((1 - n) / (1 + n)) + 4 * (n**2 + n**6) * np.log((1 + n) / (n**2 * (1 - n)))) / ((n**2 - 1)**2 * (n**2 + 1)**3)
        # equation (27)
        A = (1 + (3 / 2) * (8 * (1 - n**2)**(3/2) / (105 * n**3)) - (((n - 1)**2 * (8 + 32 * n + 52 * n**2 + 13 * n**3)) / (105 * n**3 * (1 + n)**2) + r1 + r2 + r3)) / (1 - (-3 + 7 * n + 13 * n**2 + 9 * n**3 - 7 * n**4 + 3 * n**5 + n**6 + n**7) / (3 * (n - 1) * (n + 1)**2 * (n**2 + 1)**2) - r4)
    return A

def A_fit(ne, ni):
    """
    Computes parameter A; equation (27)
    Args:
    ne: external medium
    ni: diffusing medium
    Parameter A is computed using the solution presented in:
    Contini D, Martelli F, Zaccanti G.
    Photon migration through a turbid slab described by a model
    based diffusion approximation. I. Theory
    Applied Optics Vol 36, No 19, 1997, pp 4587-4599
    """
    # page 4590
    n = ni / ne
    if n == 1:
        # page 4591
        A = 1
    elif n > 1:
        # equation (A3)
        A = 504.332889 - 2641.00214*n + 5923.699064*n**2 -7376.355814*n**3 + 5507.53041*n**4 - 2463.357945*n**5 + 610.956547*n**6 - 64.8047*n**7
    else:
        # equation (A2)
        A = 3.084635 - 6.531194*n + 8.357854*n**2 - 5.082751*n**3 + 1.171382*n**4
    return A

def f_DE(z, times, mu_s, z_s=None, A=1, v=3e10, m_tot=200):
    """
    Computes probability density function f_DE(z,t) 
    as in article "There is plenty of light at the bottom"

    Args:
    z: depths scale
    times: time scale
    mu_s: reduced scattering coefficient
    z_s: source position (default = 1/mu_s)
    A: refractive index mismatch parameter (default = 1)
    v: speed of light (default = 3e10 cm/s)
    m_tot: number of elements to account in the summation (default = 200)
    """
    if z_s is None:
        z_s = 1/mu_s
    res = np.zeros((len(times),len(z)))
    D = 1/(3*mu_s)
    z_e = 2*A*D
    i=0
    for t in times:
        _4Dvt = 4*D*v*t
        for m in range(-m_tot, m_tot+1):
            M = -2*m*z_e - 2*m*2*D - z_s
            N = -2*m*2*D - (2*m-2)*z_e +z_s
            z3mp = -2*m*z+M
            z4mp = -2*m*z+N
            a3 = -2*m+4*m*z3mp**2/_4Dvt
            a4 = -2*m+4*m*z4mp**2/_4Dvt
            print(m, M, N)
            res[i] += a3*exp(-z3mp**2/_4Dvt)-a4*exp(-z4mp**2/_4Dvt)
        res[i] = res[i]/(-z_s*exp(-(z_s)**2/_4Dvt)-(2*z_e+z_s)*exp(-(2*z_e+z_s)**2/_4Dvt))
        i+=1
    return res 

def f_DE_rho(z, rhos, mu_a, mu_s, z_s=None, A=1, v=3e10, m_tot=200):
    """
    Computes probability density function f_DE(z,rho) 
    as in article "There is plenty of light at the bottom"

    Args:
    z: depths scale
    rho: source-det distances scale
    mu_a: absorption coefficient
    mu_s: reduced scattering coefficient
    z_s: source position (default = 1/mu_s)
    A: refractive index mismatch parameter (default = 1)
    v: speed of light (default = 3e10 cm/s)
    m_tot: number of elements to account in the summation (default = 200)
    """
    if z_s is None:
        z_s = 1/mu_s
    D = 1/(3*mu_s)
    mu_eff = np.sqrt(mu_a/D)
    z_e = 2*A*D
    z0m = -2*z_e - z_s
    res = np.zeros((len(rhos),len(z)))
    for k in range(len(rhos)) :
        R_DE = 1/(4*pi)*((z_s*mu_eff*exp(-mu_eff*sqrt(rhos[k]**2+z_s**2)))/(rhos[k]**2+z_s**2)+(z_s*exp(-mu_eff*sqrt(rhos[k]**2+z_s**2)))/(rhos[k]**2+z_s**2)**(3/2)-(z0m*mu_eff*exp(-mu_eff*sqrt(rhos[k]**2+z0m**2)))/(rhos[k]**2+z0m**2)-(z0m*exp(-mu_eff*sqrt(rhos[k]**2+z0m**2)))/(rhos[k]**2+z0m**2)**(3/2))
        for m in range(-m_tot,m_tot+1):
            M = -2*m*z_e - 2*m*2*D - z_s
            N = -2*m*2*D - (2*m-2)*z_e +z_s
            zmp = 2*m*z - M
            zmm = 2*m*z - N
            Zmp = rhos[k]**2 + zmp**2
            Zmm = rhos[k]**2 + zmm**2
            res[k] += m*(exp(-mu_eff*sqrt(Zmm))*(6*mu_eff*zmm**2/Zmm**2 - 2*mu_eff/Zmm + 2*(mu_eff**2*zmm**2-1)/(Zmm**(3/2)) + 6*zmm**2/Zmm**(5/2)) - exp(-mu_eff*sqrt(Zmp))*(6*mu_eff*zmp**2/Zmp**2-2*mu_eff/Zmp+2*(mu_eff**2*zmp**2-1)/(Zmp**(3/2))+6*zmp**2/Zmp**(5/2)))
            print(m)
        res[k] = res[k] * 1/(4*pi*R_DE)
    return res

def z_max_t(t, mu_s, z_s=None, A=1, v=3e10, m_tot=200):
    """
    Computes average maximum depth reached 
    as in article "There is plenty of light at the bottom"

    Args:
    t: time scale
    mu_s: reduced scattering coefficient
    z_s: source position (default = 1/mu_s)
    A: refractive index mismatch parameter (default = 1)
    v: speed of light (default = 3e10 cm/s)
    m_tot: number of elements to account in the summation (default = 200)
    """
    if z_s is None:
        z_s = 1/mu_s
    res = np.zeros(t.shape)
    D = 1/(3*mu_s)
    z_e = 2*A*D
    _4Dvt = 4*D*v*t
    for m in range(-m_tot, m_tot+1):
        if m == 0:
            continue
        M = -2*m*z_e - 2*m*2*D - z_s
        N = -2*m*2*D - (2*m-2)*z_e + z_s
        print(m, M, N)
        res += (D*t*v/m)*(exp(-M**2/_4Dvt)-exp(-N**2/_4Dvt))
    res = res/(-z_s*exp(-(z_s)**2/_4Dvt)-(2*z_e+z_s)*exp(-(2*z_e+z_s)**2/_4Dvt))
    return res

def z_max_rho(rhos, mu_a, mu_s, z_s=None, A=1, s0=50, v=3e10, m_tot=200):
    """
    Computes average maximum depth reached 
    as in article "There is plenty of light at the bottom"

    Args:
    rhos: source-det distance scale
    mu_s: reduced scattering coefficient
    z_s: source position (default = 1/mu_s)
    A: refractive index mismatch parameter (default = 1)
    s0: thickness of the slab 
    v: speed of light (default = 3e10 cm/s)
    m_tot: number of elements to account in the summation (default = 200)
    """
    if z_s is None:
        z_s = 1/mu_s
    D = 1/(3*mu_s)
    mu_eff = np.sqrt(mu_a/D)
    z_e = 2*A*D
    z0m = -2*z_e - z_s
    z00p = z_s
    z00m = -2*z_e - z_s
    R_DE = 1/(4*pi)*((z_s*mu_eff*exp(-mu_eff*sqrt(rhos**2+z_s**2)))/(rhos**2+z_s**2)+(z_s*exp(-mu_eff*sqrt(rhos**2+z_s**2)))/(rhos**2+z_s**2)**(3/2)-(z0m*mu_eff*exp(-mu_eff*sqrt(rhos**2+z0m**2)))/(rhos**2+z0m**2)-(z0m*exp(-mu_eff*sqrt(rhos**2+z0m**2)))/(rhos**2+z0m**2)**(3/2))
    s0 = 50
    summ = np.zeros(len(rhos))
    for m in range(-m_tot,m_tot+1):
        if m==0:
            continue
        else:
            M = -2*m*z_e - 2*m*2*D - z_s
            N = -2*m*2*D - (2*m-2)*z_e +z_s
            zm0p = 2*m*s0 - M
            zm0m = 2*m*s0 - N
            summ += exp(-mu_eff*sqrt(rhos**2+zm0p**2))/(2*m*sqrt(rhos**2+zm0p**2))-exp(-mu_eff*sqrt(rhos**2+zm0m**2))/(2*m*sqrt(rhos**2+zm0m**2))-exp(-mu_eff*sqrt(rhos**2+M**2))/(2*m*sqrt(rhos**2+M**2))+exp(-mu_eff*sqrt(rhos**2+N**2))/(2*m*sqrt(rhos**2+N**2))
    res = s0 + 1/(4*pi*R_DE)*summ - s0/(4*pi*R_DE)*(z00p*mu_eff*exp(-mu_eff*sqrt(rhos**2+z00p**2))/(rhos**2+z00p**2)+z00p*exp(-mu_eff*sqrt(rhos**2+z00p**2))/(rhos**2+z00p**2)**(3/2)-z00m*mu_eff*exp(-mu_eff*sqrt(rhos**2+z00m**2))/(rhos**2+z00m**2)-z00m*exp(-mu_eff*sqrt(rhos**2+z00m**2))/(rhos**2+z00m**2)**(3/2))
    return res
# 1. Compute dR
def compute_dR(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths):
    dR = np.zeros((len(rhos), len(times), len(depths)))
    for k, rho in enumerate(rhos):
        print(f'Computing dR for rho = {rho}')
        for i, t in enumerate(times):
            for j, d in enumerate(depths):
                prefactor = mur / pi**2 * sqrt(v*ve / (16*D*De)) * exp(-muae * ve * t)
                integrand = lambda tau: (
                    exp(-(mua*v - muae*ve + mur*v) * tau)
                    / sqrt(tau * (t - tau))
                    / (4 * De * ve * (t - tau) + 4 * D * v * tau)
                    * exp(-(rho**2) / (4 * De * ve * (t - tau) + 4 * D * v * tau))
                    * (exp(-(d - zs)**2 / (4 * D * v * tau)) - exp(-(d + zs + 2 * ze)**2 / (4 * D * v * tau)))
                    * (exp(-d**2 / (4 * De * ve * (t - tau))) - exp(-(d + 2 * zee)**2 / (4 * De * ve * (t - tau))))
                )
                temp = quad(integrand, 0, t, limit=200, epsabs=1e-6, epsrel=1e-6)
                dR[k, i, j] = prefactor * temp[0]
    return dR

# 2. Compute R
def compute_R(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times):
    R = np.zeros((len(rhos), len(times), 1))
    for k, rho in enumerate(rhos):
        print(f'Computing R for rho = {rho}')
        for i, t in enumerate(times):
            prefactor = mur / pi**2 * sqrt(v*ve / (16*D*De)) * exp(-muae * ve * t)
            conv_raman_semi = lambda tau, z: (
                exp(-(mua*v - muae*ve + mur*v) * tau)
                / sqrt(tau * (t - tau))
                / (4 * De * ve * (t - tau) + 4 * D * v * tau)
                * exp(-(rho**2) / (4 * De * ve * (t - tau) + 4 * D * v * tau))
                * (exp(-(z - zs)**2 / (4 * D * v * tau)) - exp(-(z + zs + 2 * ze)**2 / (4 * D * v * tau)))
                * (exp(-z**2 / (4 * De * ve * (t - tau))) - exp(-(z + 2 * zee)**2 / (4 * De * ve * (t - tau))))
            )
            temp = dblquad(conv_raman_semi, 0, np.inf, 0, t, epsabs=1e-4, epsrel=1e-4)
            R[k, i] = prefactor * temp[0]
    return R

# 3. Compute PDF
def compute_pdf(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths):
    dR = compute_dR(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths)
    R = compute_R(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times)
    prob = np.zeros_like(dR)
    for j in range(len(R)):
        for k in range(len(R[j])):
            if R[j, k, 0] > 0:
                prob[j, k] = dR[j, k] / R[j, k, 0]
            else:
                prob[j, k] = 0
    return prob

def compute_pdf_cw(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths):
    dR = compute_dR(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times, depths)
    R = compute_R(mur, v, ve, D, De, muae, mua, musp, zs, ze, zee, rhos, times)
    dR_cw = trapz(dR, times, axis=1)
    R_cw = trapz(R, times, axis=1)
    prob_cw = np.zeros_like(dR_cw)
    for k in range(len(R_cw)):
        if R_cw[k] > 0:
            prob_cw[k] = dR_cw[k] / R_cw[k]
        else:
            prob_cw[k] = 0
    return prob_cw

# 4. Compute average depth
def compute_avg_depth(prob, depths):
    avg_depth = np.zeros((prob.shape[0], prob.shape[1]))
    for j in range(prob.shape[0]):
        for k in range(prob.shape[1]):
            avg_depth[j, k] = trapz(depths * prob[j, k], depths)
    return avg_depth

def compute_avg_depth_cw(prob_cw, depths):
    avg_depth_cw = np.zeros(prob_cw.shape[0])
    for k in range(prob_cw.shape[0]):
        avg_depth_cw[k] = trapz(depths * prob_cw[k], depths)
    return avg_depth_cw

# 5. Compute standard deviation of depth
def compute_sigma_avg_depth(prob, depths, avg_depth):
    sigma_avg_depth = np.zeros(avg_depth.shape)
    for j in range(prob.shape[0]):
        for k in range(prob.shape[1]):
            variance = trapz((depths - avg_depth[j, k])**2 * prob[j, k], depths)
            sigma_avg_depth[j, k] = sqrt(variance)
    return sigma_avg_depth

def compute_sigma_avg_depth_cw(prob_cw, depths, avg_depth_cw):
    sigma_avg_depth_cw = np.zeros(avg_depth_cw.shape)
    for k in range(prob_cw.shape[0]):
        variance = trapz((depths - avg_depth_cw[k])**2 * prob_cw[k], depths)
        sigma_avg_depth_cw[k] = sqrt(variance)
    return sigma_avg_depth_cw

