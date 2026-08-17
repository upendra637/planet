
import numpy as np
import matplotlib.pyplot as plt

try:
    import emcee
    HAVE_EMCEE = True
except ImportError:
    HAVE_EMCEE = False


# ============================================================
# RAPPAPORT ET AL. (2018) EXOCOMET MODEL
# KIC 3542116
# ============================================================
#
# This is a reconstruction of the model described in Section 6
# of Rappaport et al. (2018), "Likely transiting exocomets
# detected by Kepler".
#
# The paper states:
#   - quadratic limb darkening
#   - exponential dust-tail optical depth
#   - comet position xc = vt (t - t0)
#   - six fitted parameters: t0, lambda, C, vt, b, DC
#   - numerical integration at 6-min resolution
#   - convolution with Kepler's 30-min long-cadence integration
#   - Metropolis-Hastings MCMC
#
# The original historical source code was not supplied with the
# paper, so this is a Python reconstruction, not a copy of their
# original code.
#
# Units:
#   time       : days
#   x, y       : R_star
#   lambda     : R_star
#   b          : R_star
#   tail_width : R_star
#   vt         : R_star/day
#   C          : dimensionless optical depth
#   DC         : flux normalization
# ============================================================


# ------------------------------------------------------------
# 1. Quadratic limb darkening
# ------------------------------------------------------------

def quadratic_limb_darkening(mu, u1=0.30, u2=0.20):
    """
    I(mu)/I(1) = 1 - u1(1-mu) - u2(1-mu)^2
    """
    return 1.0 - u1 * (1.0 - mu) - u2 * (1.0 - mu)**2


# ------------------------------------------------------------
# 2. Stellar disk
# ------------------------------------------------------------

def make_star_grid(n_grid=500, u1=0.30, u2=0.20):
    """
    Construct a Cartesian stellar disk in units of R_star.
    """

    x = np.linspace(-1.0, 1.0, n_grid)
    y = np.linspace(-1.0, 1.0, n_grid)

    X, Y = np.meshgrid(x, y)

    r2 = X**2 + Y**2
    disk = r2 <= 1.0

    mu = np.zeros_like(X)
    mu[disk] = np.sqrt(1.0 - r2[disk])

    intensity = np.zeros_like(X)
    intensity[disk] = quadratic_limb_darkening(
        mu[disk], u1, u2
    )

    F_star = np.sum(intensity)

    return X, Y, disk, intensity, F_star


# ------------------------------------------------------------
# 3. Exponential dust-tail optical depth
# ------------------------------------------------------------

def comet_optical_depth(
    X,
    Y,
    xc,
    C,
    lam,
    b,
    tail_width
):
    """
    Rappaport et al. optical-depth model.

    tau = C exp[-(xc-x)/lambda]

    when:
        x < xc
        |y-b| <= tail_width/2

    Otherwise:
        tau = 0
    """

    tau = np.zeros_like(X)

    region = (
        (X < xc)
        & (np.abs(Y - b) <= tail_width / 2.0)
    )

    tau[region] = (
        C
        * np.exp(
            -(xc - X[region]) / lam
        )
    )

    return tau


# ------------------------------------------------------------
# 4. Instantaneous flux
# ------------------------------------------------------------

def instantaneous_flux(
    X,
    Y,
    intensity,
    F_star,
    xc,
    C,
    lam,
    b,
    tail_width
):
    """
    Integrate the attenuated stellar intensity.

    I_observed = I_star exp(-tau)

    F/F0 = sum[I_observed] / sum[I_star]
    """

    tau = comet_optical_depth(
        X, Y,
        xc=xc,
        C=C,
        lam=lam,
        b=b,
        tail_width=tail_width
    )

    attenuated = intensity * np.exp(-tau)

    return np.sum(attenuated) / F_star


# ------------------------------------------------------------
# 5. Six-minute model
# ------------------------------------------------------------

def high_resolution_model(
    time,
    t0,
    lam,
    C,
    vt,
    b,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=500
):
    """
    Generate the model at 6-minute internal resolution.

    Equation:
        xc = vt (t - t0)
    """

    X, Y, disk, intensity, F_star = make_star_grid(
        n_grid=n_grid,
        u1=u1,
        u2=u2
    )

    dt = 6.0 / (24.0 * 60.0)

    tmin = np.min(time)
    tmax = np.max(time)

    # Padding prevents convolution edge effects.
    pad = 0.5

    t_internal = np.arange(
        tmin - pad,
        tmax + pad + dt,
        dt
    )

    flux_internal = np.empty_like(t_internal)

    for i, t in enumerate(t_internal):

        # Rappaport et al. equation (3)
        xc = vt * (t - t0)

        flux_internal[i] = instantaneous_flux(
            X, Y,
            intensity,
            F_star,
            xc=xc,
            C=C,
            lam=lam,
            b=b,
            tail_width=tail_width
        )

    return t_internal, flux_internal


# ------------------------------------------------------------
# 6. Kepler long-cadence integration
# ------------------------------------------------------------

def kepler_long_cadence_model(
    time,
    t0,
    lam,
    C,
    vt,
    b,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=500
):
    """
    Convolve the 6-minute model with the 30-minute
    Kepler long-cadence integration.
    """

    t_internal, flux_internal = high_resolution_model(
        time,
        t0=t0,
        lam=lam,
        C=C,
        vt=vt,
        b=b,
        tail_width=tail_width,
        u1=u1,
        u2=u2,
        n_grid=n_grid
    )

    # Five 6-minute samples = 30 minutes.
    kernel = np.ones(5) / 5.0

    convolved = np.convolve(
        flux_internal,
        kernel,
        mode="same"
    )

    return np.interp(
        time,
        t_internal,
        convolved
    )


# ------------------------------------------------------------
# 7. Complete six-parameter model
# ------------------------------------------------------------

def comet_model(
    time,
    t0,
    lam,
    C,
    vt,
    b,
    DC,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=500
):
    """
    Six free parameters:

        t0
        lambda
        C
        vt
        b
        DC
    """

    relative_flux = kepler_long_cadence_model(
        time,
        t0=t0,
        lam=lam,
        C=C,
        vt=vt,
        b=b,
        tail_width=tail_width,
        u1=u1,
        u2=u2,
        n_grid=n_grid
    )

    return DC * relative_flux


# ------------------------------------------------------------
# 8. Log-prior
# ------------------------------------------------------------

def log_prior(theta):
    """
    Broad example priors.

    theta = [t0, lambda, C, vt, b, DC]

    Adjust these for the actual transit being fitted.
    """

    t0, lam, C, vt, b, DC = theta

    if not (-2.0 < t0 < 2.0):
        return -np.inf

    if not (0.01 < lam < 3.0):
        return -np.inf

    if not (1e-6 < C < 1.0):
        return -np.inf

    if not (0.1 < vt < 20.0):
        return -np.inf

    if not (-1.0 < b < 1.0):
        return -np.inf

    if not (0.8 < DC < 1.2):
        return -np.inf

    return 0.0


# ------------------------------------------------------------
# 9. Log-likelihood
# ------------------------------------------------------------

def log_likelihood(
    theta,
    time,
    flux,
    flux_err,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=300
):
    """
    Gaussian likelihood based on chi-square.
    """

    t0, lam, C, vt, b, DC = theta

    model = comet_model(
        time,
        t0=t0,
        lam=lam,
        C=C,
        vt=vt,
        b=b,
        DC=DC,
        tail_width=tail_width,
        u1=u1,
        u2=u2,
        n_grid=n_grid
    )

    residual = flux - model

    chi2 = np.sum(
        (residual / flux_err)**2
    )

    return -0.5 * chi2


# ------------------------------------------------------------
# 10. Posterior
# ------------------------------------------------------------

def log_probability(
    theta,
    time,
    flux,
    flux_err,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=300
):

    lp = log_prior(theta)

    if not np.isfinite(lp):
        return -np.inf

    return lp + log_likelihood(
        theta,
        time,
        flux,
        flux_err,
        tail_width=tail_width,
        u1=u1,
        u2=u2,
        n_grid=n_grid
    )


# ------------------------------------------------------------
# 11. emcee fitting
# ------------------------------------------------------------

def run_emcee(
    time,
    flux,
    flux_err,
    initial,
    n_walkers=24,
    n_steps=2000,
    tail_width=0.20,
    u1=0.30,
    u2=0.20,
    n_grid=250
):
    """
    Fit the six-parameter model using emcee.
    """

    if not HAVE_EMCEE:
        raise ImportError(
            "Install emcee first: pip install emcee"
        )

    initial = np.asarray(
        initial,
        dtype=float
    )

    ndim = 6

    scales = np.array([
        0.005,    # t0
        0.02,     # lambda
        0.0005,   # C
        0.05,     # vt
        0.02,     # b
        0.001     # DC
    ])

    p0 = (
        initial
        + scales * np.random.randn(
            n_walkers,
            ndim
        )
    )

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_probability,
        args=(
            time,
            flux,
            flux_err,
            tail_width,
            u1,
            u2,
            n_grid
        )
    )

    sampler.run_mcmc(
        p0,
        n_steps,
        progress=True
    )

    return sampler


# ------------------------------------------------------------
# 12. Example synthetic transit
# ------------------------------------------------------------

if __name__ == "__main__":

    # D1268-like parameters from Table 2.
    #
    # Paper:
    # vt     = 3.70 R_star/day
    # lambda = 0.72 R_star
    # b      = 0.27 R_star
    #
    # C is not listed in Table 2, so the value below is
    # illustrative.
    #
    # tail_width is degenerate with C in the paper, so it is
    # fixed rather than fitted.

    true_params = [
        0.0,       # t0
        0.72,      # lambda
        0.02,      # C
        3.70,      # vt [R_star/day]
        0.27,      # b [R_star]
        1.0        # DC
    ]

    t0, lam, C, vt, b, DC = true_params

    # Three days around the event.
    cadence = 30.0 / (24.0 * 60.0)

    time = np.arange(
        -1.5,
        1.5,
        cadence
    )

    print("Generating synthetic exocomet transit...")

    flux = comet_model(
        time,
        t0=t0,
        lam=lam,
        C=C,
        vt=vt,
        b=b,
        DC=DC,
        tail_width=0.20,
        u1=0.30,
        u2=0.20,
        n_grid=500
    )

    # Add representative Kepler-like Gaussian noise.
    rng = np.random.default_rng(1234)

    flux_err = np.ones_like(flux) * 35e-6

    noisy_flux = (
        flux
        + rng.normal(
            0.0,
            flux_err
        )
    )

    # --------------------------------------------------------
    # Plot synthetic data and true model
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))

    plt.errorbar(
        time,
        noisy_flux,
        yerr=flux_err,
        fmt=".",
        ms=3,
        alpha=0.5,
        label="synthetic Kepler data"
    )

    plt.plot(
        time,
        flux,
        lw=2,
        label="Rappaport model"
    )

    plt.axvline(
        t0,
        linestyle="--",
        label=r"$t_0$"
    )

    plt.xlabel("Time from transit center [days]")
    plt.ylabel("Normalized flux")
    plt.title("KIC 3542116-like Exocomet Transit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # MCMC
    # --------------------------------------------------------
    #
    # Uncomment this section to fit the synthetic data.
    #
    # initial = [
    #     0.01,
    #     0.70,
    #     0.018,
    #     3.60,
    #     0.25,
    #     1.00
    # ]
    #
    # sampler = run_emcee(
    #     time,
    #     noisy_flux,
    #     flux_err,
    #     initial,
    #     n_walkers=24,
    #     n_steps=1000
    # )
    #
    # samples = sampler.get_chain(
    #     discard=300,
    #     flat=True
    # )
    #
    # best = np.median(
    #     samples,
    #     axis=0
    # )
    #
    # names = [
    #     "t0",
    #     "lambda",
    #     "C",
    #     "vt",
    #     "b",
    #     "DC"
    # ]
    #
    # print("\nPosterior medians:")
    # for name, value in zip(names, best):
    #     print(f"{name:8s} = {value:.6g}")
    #
    # fitted = comet_model(
    #     time,
    #     *best,
    #     tail_width=0.20
    # )
    #
    # plt.figure(figsize=(10, 5))
    # plt.errorbar(
    #     time,
    #     noisy_flux,
    #     yerr=flux_err,
    #     fmt=".",
    #     ms=3,
    #     alpha=0.5,
    #     label="data"
    # )
    #
    # plt.plot(
    #     time,
    #     fitted,
    #     lw=2,
    #     label="MCMC model"
    # )
    #
    # plt.xlabel("Time [days]")
    # plt.ylabel("Normalized flux")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()