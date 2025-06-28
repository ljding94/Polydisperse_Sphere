import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

_RNORM = np.random.normal(0.0, 1.0, 10_000)     # standard normal, fixed
_RUNIF = np.random.uniform(-1.0, 1.0, 10_000)   # for uniform case



def gen_Rs_from_distribution(R0, sigma, pdType):
    if pdType == 1:                 # uniform
        return R0 * (1 + sigma * _RUNIF)
    elif pdType == 2:               # normal
        return R0 * (1 + sigma * _RNORM)
    elif pdType == 3:               # log-normal
        return R0 * np.exp(sigma * _RNORM)
    else:
        raise ValueError("Unknown polydisperse type")
'''
def gen_Rs_from_distribution(R0, sigma, pdType, n=10000):
    # 1. sample polydisperse R
    if pdType == 1:  # uniform distribution
        #Rs = R0 * np.random.uniform(1 - sigma, 1 + sigma, size=10000)
        Rs = R0 * np.random.uniform(1 - sigma, 1 + sigma, size=10000)
    elif pdType == 2:  # normal distribution
        Rs = R0 * np.random.normal(1, sigma, size=10000)
    elif pdType == 3:  # log-normal distribution
        z = np.random.normal(0, sigma, size=10000)
        Rs = R0 * np.exp(z)
    else:
        raise ValueError("Unknown polydisperse type")

    return Rs
'''

def calc_volume_fraction(R0, N, sigma, pdType):
    # use analytical formula for volume fraction
    if pdType == 1:  # uniform distribution
        # int_{1-sigma}^{1+sigma} pi/6*D**3 dD = pi/6 * (2*sigma) * R0**3}
        eta = N * 4 * np.pi / 3 * R0**3 * (1 + sigma**2)
    elif pdType == 2:  # normal distribution
        # For normal distribution: R = R0 * (1 + sigma*z) where z ~ N(0,1)
        # Volume fraction: eta = N * 4*pi/3 * <R^3>
        # <R^3> = R0^3 * (1 + 3*sigma^2)
        eta = N * 4 * np.pi / 3 * R0**3 * (1 + 3 * sigma**2)
    elif pdType == 3:  # log-normal distribution
        # <R^3> = R0^3 * exp(9/2*sigma^2)
        # https://en.wikipedia.org/wiki/Log-normal_distribution
        eta = N * 4 * np.pi / 3 * R0**3 * np.exp(4.5 * sigma**2)

    return eta


def calc_sphere_Fq_stats(Rs, Q, sigma, pdType):
    FQs = []
    Vs = 4.0 * np.pi / 3.0 * np.array(Rs) ** 3  # volume of sphere squared
    for i in range(Rs.shape[0]):
        qR = Q * Rs[i]
        FQ = Vs[i] * 3 * (np.sin(qR) - qR * np.cos(qR)) / (qR) ** 3
        FQs.append(FQ)
    mean_V2 = np.mean(Vs**2)
    mean_FQ = np.mean(FQs, axis=0)
    mean_FQ_sq = np.mean(np.array(FQs) ** 2, axis=0)

    # P(Q) = <FQ^2>/<V^2>
    return mean_V2, mean_FQ, mean_FQ_sq


def calc_HS_PY_SQ(Q, Reff, eta):
    # S_PY(Q) = 1/(1+24 phi G(x,phi)/x)

    # https://en.wikipedia.org/wiki/Percus–Yevick_approximation
    """
    /* abridged from sasmodels/models/hardsphere.c */
    double py(double qr, double eta) {
        const double a = pow(1+2*eta, 2)/pow(1-eta, 4);
        const double b = -6*eta*pow(1+eta/2, 2)/pow(1-eta, 4);
        const double c = 0.5*eta*a;
        const double x = 2*qr;                 //  x = 2 q R_eff
        const double x2 = x*x;

        const double G =
            a/x2   *(sin(x)-x*cos(x))
            + b/x2/x *(2*x*sin(x)+(2-x2)*cos(x)-2)
            + c/pow(x,5)*(-pow(x,4)*cos(x)
                + 4*((3*x2-6)*cos(x) + x*(x2-6)*sin(x) + 6));

        return 1.0/(1.0 + 24.0*eta*G/x);
    }
    """
    a = (1 + 2 * eta) ** 2 / (1 - eta) ** 4
    b = -6 * eta * (1 + eta / 2) ** 2 / (1 - eta) ** 4
    c = 0.5 * eta * a
    x = 2 * Q * Reff  # x = 2 q R_eff
    x2 = x * x

    G = (
        a / x2 * (np.sin(x) - x * np.cos(x))
        + b / x2 / x * (2 * x * np.sin(x) + (2 - x2) * np.cos(x) - 2)
        + c / x**5 * (-(x**4) * np.cos(x) + 4 * ((3 * x2 - 6) * np.cos(x) + x * (x2 - 6) * np.sin(x) + 6))
    )

    SQ_PY = 1.0 / (1.0 + 24.0 * eta * G / x)
    return SQ_PY


def calc_HS_PY_IQ(Qs, eta, sigma, pdType, beta_correction=True):
    # 1. find all Rs from distribution
    R0 = 0.5  # effective radius
    Rs = gen_Rs_from_distribution(R0, sigma, pdType)

    # find Reff, mean-volum sphere radius
    Reff = np.mean(Rs**3) ** (1 / 3)

    # 2. calculate Fq stats
    mean_V2, mean_FQ, mean_FQ_sq = calc_sphere_Fq_stats(Rs, Qs, sigma, pdType)
    PQ = mean_FQ_sq / mean_V2

    # 3. calculate effective S(Q)
    SQ_PY = calc_HS_PY_SQ(Qs, Reff, eta)

    # find the beta correction
    if beta_correction:
        beta = mean_FQ**2 / mean_FQ_sq
    else:
        beta = np.ones_like(Qs)

    # 4. calculate effective S(Q)
    SQ_eff = 1 + beta * (SQ_PY - 1)

    # 5. calculate I(Q)
    IQ = SQ_eff * PQ
    return IQ


def fit_eta_sigma(q, Iq, pdType=2, p0=(0.10, 0.05), bounds=([0.0001, 0.001], [1.0, 0.200]), absolute_sigma=False):  # experimental data  # initial guesses (eta, sigma)  # lower bounds  # upper bounds

    func = lambda q, eta, sigma: np.log10(calc_HS_PY_IQ(q, eta, sigma, pdType))

    def model(q, eta, sigma):
            y = calc_HS_PY_IQ(q, eta, sigma, pdType)
            #if np.any(~np.isfinite(y)) or np.any(y <= 0):
            #    return np.full_like(q, 1e300)
            return np.log10(y)

    popt, pcov = curve_fit(model, q, np.log10(Iq), p0=p0, bounds=bounds)  # raise if your data are tricky to fit

    print(f"Fitted parameters: eta = {popt[0]:.4f}, sigma = {popt[1]:.4f}")
    eta_fit, sigma_fit = popt
    eta_err, sigma_err = np.sqrt(np.diag(pcov))

    return {"eta": eta_fit, "sigma": sigma_fit, "eta_err": eta_err, "sigma_err": sigma_err, "cov": pcov}


# --- 3.  Example usage -------------------------------------------------------
if __name__ == "__main__":
    # q_exp, Iq_exp = np.loadtxt("my_data.dat", unpack=True)   # your data here
    # result = fit_eta_sigma(q_exp, Iq_exp, pdType="lognormal")
    # print(result)
    pass
