"""Implements Automatic Gain Control (AGC) for audio signals,
as described in http://labrosa.ee.columbia.edu/matlab/tf_agc/
"""

import numpy as np
import scipy.signal as signal

from utils import fft2melmx
from stft import stft, istft


def tf_agc(d, sr, t_scale=0.5, f_scale=1.0, causal_tracking=True, plot=False):
    """
    Perform frequency-dependent automatic gain control on an auditory
    frequency axis.
    d is the input waveform (at sampling rate sr);
    y is the output waveform with approximately constant
    energy in each time-frequency patch.
    t_scale is the "scale" for smoothing in time (default 0.5 sec).
    f_scale is the frequency "scale" (default 1.0 "mel").
    causal_tracking == 0 selects traditional infinite-attack, exponential release.
    causal_tracking == 1 selects symmetric, non-causal Gaussian-window smoothing.
    D returns actual STFT used in analysis.  E returns the
    smoothed amplitude envelope divided out of D to get gain control.
    """

    hop_size = 0.032  # in seconds

    # Make STFT on ~32 ms grid
    ftlen = int(2 ** np.round(np.log(hop_size * sr) / np.log(2.)))
    winlen = ftlen
    hoplen = winlen / 2
    D = stft(d, winlen, hoplen)  # using my code
    ftsr = sr / hoplen
    ndcols = D.shape[1]

    # Smooth in frequency on ~ mel resolution
    # Width of mel filters depends on how many you ask for,
    # so ask for fewer for larger f_scales
    nbands = max(10, 20 / f_scale)  # 10 bands, or more for very fine f_scale
    mwidth = f_scale * nbands / 10  # will be 2.0 for small f_scale
    (f2a_tmp, _) = fft2melmx(ftlen, sr, int(nbands), mwidth)
    f2a = f2a_tmp[:, :ftlen / 2 + 1]
    audgram = np.dot(f2a, np.abs(D))

    if causal_tracking:
        # traditional attack/decay smoothing
        fbg = np.zeros(audgram.shape)
        # state = zeros(size(audgram,1),1);
        state = np.zeros(audgram.shape[0])
        alpha = np.exp(-(1. / ftsr) / t_scale)
        for i in range(audgram.shape[1]):
            state = np.maximum(alpha * state, audgram[:, i])
            fbg[:, i] = state

    else:
        # noncausal, time-symmetric smoothing
        # Smooth in time with tapered window of duration ~ t_scale
        tsd = np.round(t_scale * ftsr) / 2
        htlen = 6 * tsd  # Go out to 6 sigma
        twin = np.exp(-0.5 * (((np.arange(-htlen, htlen + 1)) / tsd) ** 2)).T

        # reflect ends to get smooth stuff
        AD = audgram
        x = np.hstack((np.fliplr(AD[:, :htlen]),
                       AD,
                       np.fliplr(AD[:, -htlen:]),
                       np.zeros((AD.shape[0], htlen))))
        fbg = signal.lfilter(twin, 1, x, 1)

        # strip "warm up" points
        fbg = fbg[:, twin.size + np.arange(ndcols)]

    # map back to FFT grid, flatten bark loop gain
    sf2a = np.sum(f2a, 0)
    sf2a_fix = sf2a
    sf2a_fix[sf2a == 0] = 1.
    E = np.dot(np.dot(np.diag(1. / sf2a_fix), f2a.T), fbg)
    # Remove any zeros in E (shouldn't be any, but who knows?)
    E[E <= 0] = np.min(E[E > 0])

    # invert back to waveform
    y = istft(D / E, winlen, hoplen, window=np.ones(winlen))  # using my code

    if plot:
        try:
            import matplotlib.pyplot as plt
            plt.subplot(3, 1, 1)
            plt.imshow(20. * np.log10(np.flipud(np.abs(D))))
            plt.subplot(3, 1, 2)
            plt.imshow(20. * np.log10(np.flipud(np.abs(E))))
            A = stft(y, winlen, hoplen)  # using my code
            plt.subplot(3, 1, 3)
            plt.imshow(20. * np.log10(np.flipud(np.abs(A))))
            plt.show()
        except Exception, e:
            print "Failed to plot results"
            print e

    return (y, D, E)
