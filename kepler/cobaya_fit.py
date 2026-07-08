import batman
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from bin import flatNbin
import matplotlib.pyplot as plt
from cobaya.run import run

binn = flatNbin('KIC 6922244', 'kepler', 'long', 4)
flc = binn.flat_data(401)
pg = binn.bls_method(1, 100, 10000)

period = float(pg.period_at_max_power.value)
transit_time= float(pg.transit_time_at_max_power.value)
flux = flc.flux.value
time = flc.time.value
flux_err = flc.flux_err.value


def transit_model(theta):

    t0, per, rp, a, inc = theta

    params = batman.TransitParams()

    params.t0 = t0
    params.per = per
    params.rp = rp
    params.a = a
    params.inc = inc

    params.ecc = 0.0
    params.w = 90.0

    params.u = [0.3, 0.2]
    params.limb_dark = "quadratic"

    m = batman.TransitModel(params, time)

    return m.light_curve(params)

def loglike(t0, per, rp, a, inc):
    
    model = transit_model(
        [t0, per, rp, a, inc]
    )

    chi2 = np.sum(
        ((flux - model)/flux_err)**2
    )

    return float(-0.5 * chi2)

    

try:

    info = {

        "likelihood": {
            "transit": {
                "external": loglike,
                "input_params": [
                    "t0",
                    "per",
                    "rp",
                    "a",
                    "inc"
                ]
                
            }
        },

        "params": {

            "t0": {
                "prior": {
                    "min": 352.0000,
                    "max": 360.0000
                },
                "ref": float(transit_time),
                "proposal": 0.0002
            },

            "per": {
                "prior": {
                    "min": 3.4500,
                    "max": 3.5500
                },
                "ref": float(period),
                "proposal": 1e-4
            },

            "rp": {
                "prior": {
                    "min": 0.0100,
                    "max": 0.2000 
                },
                "ref": 0.1,
                "proposal": 0.005
            },

            "a": {
                "prior": {
                    "min": 1.0000,
                    "max": 8.0000
                },
                "ref": 3.0,
                "proposal": 0.02
            },

            "inc": {
                "prior": {
                    "min": 70.0000,
                    "max": 90.0000
                },
                "ref": 74.0,
                "proposal": 0.1
            }
        },

        "sampler": {
            "mcmc": {
                "Rminus1_stop": 0.001,
                "max_tries": 10000
            }
        }
    }

    updated_info, sampler = run(info)

    samples = sampler.products()["sample"]

    print(type(samples))

    best = samples.mean()

    print(best)

    best_t0  = best[0]
    best_per = best[1]
    best_rp  = best[2]
    best_a   = best[3]
    best_inc = best[4]

    best_model = transit_model(
        [best_t0,
        best_per,
        best_rp,
        best_a,
        best_inc]
    )



    plt.figure(figsize=(8,5))

    plt.scatter(
        time,
        flux,
        s=5,
        color="black",
        label="Data"
    )

    plt.plot(
        time,
        best_model,
        color="red",
        lw=2,
        label="BATMAN fit"
    )

    plt.legend()
    plt.show()

except Exception as e:
    print("An error occurred:", e)
    

