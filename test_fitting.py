import numpy as np
import mdreg

def dwi_pixel_model(bVal, S0, ADC):
    return np.abs(S0 * np.exp(-bVal * ADC))

def dwi_pixel_model_init(bVal, signal):
    bVal_neg = - bVal.reshape(2,1)
    A = np.append(bVal_neg, np.ones(bVal_neg.shape), axis=1)
    B = np.log(signal.reshape(2,1))
    X = np.linalg.solve(A, B)
    ADC = X[0].item()
    S0 = np.exp(X[1].item())
    return [S0, ADC]

data = mdreg.fetch('MOLLI')

# We will consider the slice z=0 of the data array:
array = data['array']

# Use the built-in animation function of mdreg to visualise the motion:
# mdreg.animation(array, vmin=0, vmax=1e4, show=True)

molli = {

    # The function to fit the data
    'func': mdreg.abs_exp_recovery_2p,

    # The keyword arguments required by the function
    'TI': np.array(data['TI'])/1000,
}

# Perform model-driven coregistration
coreg, defo, fit, pars = mdreg.fit(array, fit_image=molli)

# Visualise the results
mdreg.plot_series(array, fit, coreg, vmin=0, vmax=1e4, show=True)


dwi_pixel_fit = {
    # The custom-built single pixel model
    'model': dwi_pixel_model,

    # xdata for the single-pixel model
    'xdata': np.array(data['bVal'])*1e-6,

    # Optional: custom-built initialization function
    'func_init': dwi_pixel_model_init,

    # Optional: bounds for the free model parameters
    'bounds': (0, np.inf),
}
