import numpy as np
import astropy.io.fits as fits

f = fits.open('v6_0_4/spec-15143-59205-04544940698.fits')

np.save('dataset/data.npy', {'loglam' : f['loglam'], 'x': 10 ** f['loglam'], 'count': 200000})