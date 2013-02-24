import numpy as np


def fft2melmx(nfft, sr=8000., nfilts=None, width=1., minfrq=0., maxfrq=None, htkmel=False, constamp=False):
    """
    Generate a matrix of weights to combine FFT bins into Mel
    bins.  nfft defines the source FFT size at sampling rate sr.
    Optional nfilts specifies the number of output bands required
    (else one per "mel/width"), and width is the constant width of each
    band relative to standard Mel (default 1).
    While wts has nfft columns, the second half are all zero.
    Hence, Mel spectrum is fft2melmx(nfft,sr)*abs(fft(xincols,nfft));
    minfrq is the frequency (in Hz) of the lowest band edge;
    default is 0, but 133.33 is a common standard (to skip LF).
    maxfrq is frequency in Hz of upper edge; default sr/2.
    You can exactly duplicate the mel matrix in Slaney's mfcc.m
    as fft2melmx(512, 8000, 40, 1, 133.33, 6855.5, 0);
    htkmel=1 means use HTK's version of the mel curve, not Slaney's.
    constamp=1 means make integration windows peak at 1, not sum to 1.
    frqs returns bin center frqs.
    """

    if maxfrq is None:
        maxfrq = sr / 2.

    if nfilts is None:
        nfilts = int(np.ceil(hz2mel(maxfrq, htkmel) / 2.))

    wts = np.zeros((nfilts, nfft))

    # Center freqs of each FFT bin
    fftfrqs = np.arange(nfft / 2 + 1, dtype=float) / nfft * sr

    # 'Center freqs' of mel bands - uniformly spaced between limits
    minmel = hz2mel(minfrq, htkmel)
    maxmel = hz2mel(maxfrq, htkmel)
    binfrqs = mel2hz(minmel + np.arange(nfilts + 2, dtype=float) / (nfilts + 1) * (maxmel - minmel), htkmel)

    # binbin = round(binfrqs / sr * (nfft - 1))

    for i in range(nfilts):
        fs = binfrqs[i + np.array([0, 1, 2])]
        # scale by width
        fs = fs[1] + width * (fs - fs[1])
        # lower and upper slopes for all bin
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        # .. then intersect them with each other and zero
        wts[i, :nfft / 2 + 1] = np.maximum(0, np.minimum(loslope, hislope))

    if not constamp:
        # Slaney-style mel is scaled to be approx constant E per channel
        wts = np.dot(np.diag(2. / (binfrqs[2:nfilts + 2] - binfrqs[:nfilts])), wts)

    # Make sure 2nd half of FFT is zero
    wts[:, (nfft / 2 + 2):] = 0
    # seems like a good idea to avoid aliasing

    return (wts, binfrqs)


def mel2hz(z, htk=False):
    """
    Convert 'mel scale' frequencies into Hz.
    Optional htk=True means use the HTK formula; else use the formula from Malcolm Slaney's mfcc.m
    """

    if htk:
        f = 700. * (10. ** (z / 2595.) - 1)
    else:
        f_0 = 0  # 133.33333
        f_sp = 200. / 3.  # 66.66667
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp  # starting mel value for log region
        logstep = np.exp(np.log(6.4) / 27.)  # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        linpts = (z < brkpt)

        f = 0 * z

        if np.isscalar(z):
            f = f_0 + f_sp * z if linpts else brkfrq * np.exp(np.log(logstep) * (z - brkpt))
        else:
            # fill in parts separately
            f[linpts] = f_0 + f_sp * z[linpts]
            f[~linpts] = brkfrq * np.exp(np.log(logstep) * (z[~linpts] - brkpt))

    return f


def hz2mel(f, htk=False):
    """
    Convert frequencies f (in Hz) to mel 'scale'.
    Optional htk=True uses the mel axis defined in the HTKBook; otherwise use Malcolm Slaney's formula.
    """

    if htk:
        z = 2595. * np.log10(1. + f / 700.)
    else:
        # pass
        f_0 = 0  # 133.33333;
        f_sp = 200. / 3.  # 66.66667;
        brkfrq = 1000.
        brkpt = (brkfrq - f_0) / f_sp  # starting mel value for log region
        logstep = np.exp(np.log(6.4) / 27.)  # the magic 1.0711703 which is the ratio needed to get from 1000 Hz to 6400 Hz in 27 steps, and is *almost* the ratio between 1000 Hz and the preceding linear filter center at 933.33333 Hz (actually 1000/933.33333 = 1.07142857142857 and  exp(log(6.4)/27) = 1.07117028749447)

        linpts = (f < brkfrq)

        z = 0 * f

        if np.isscalar(f):
            z = (f - f_0) / f_sp if linpts else brkpt + (np.log(f / brkfrq)) / np.log(logstep)
        else:
            # fill in parts separately
            z[linpts] = (f[linpts] - f_0) / f_sp
            z[~linpts] = brkpt + (np.log(f[~linpts] / brkfrq)) / np.log(logstep)

    return z


if __name__ == '__main__':

    from scipy.io import savemat
    for i in (5, 6, 7):
        n = 2 ** i
        wts, binfrqs = fft2melmx(n)
        savemat('../f2m%d.mat' % (n,), {'wts': wts, 'binfrqs': binfrqs})




# Testing values
"""

hz2mel:

f = [100,300,500,1000,5000,10000];

hz2mel(f,0)
ans = 1.5000    4.5000    7.5000   15.0000   38.4094   48.4913

hz2mel(f,1)
ans = 150.49    401.97    607.45    999.99   2363.47   3073.22


mel2hz:

z = [1.5, 4.5, 7.5, 15, 40, 48]

mel2hz(z,0)
ans = 100.00    300.00    500.00   1000.00   5577.80   9667.88

mel2hz(z,1)
ans = 0.93230    2.80063    4.67394    9.37910   25.29102   30.45783

"""