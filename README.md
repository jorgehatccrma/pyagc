Automatic Gain Control
======================

 * Author: Jorge Herrera (jorgeh@ccrma.stanford.edu)


Code that implements Automatic Gain Control (AGC) for audio signals in python, based in [Dan Ellis' code][ref1]

The code is based in the original Matlab implementation in the above link, with some minor *pythonizations*. A sample `WAV` file is provided. The file was obtained from the original Matlab source code.

## Dependencies

The code depends on NumPy and SciPy.

## Example of usage

    import scipy.io.wavfile
    import numpy as np
    from agc import tf_agc

    # read audiofile
    sr, d = scipy.io.wavfile.read('speech.wav')

    # convert from int16 to float (-1,1) range
    convert_16_bit = float(2 ** 15)
    d = d / (convert_16_bit + 1.0)

    # apply AGC
    (y, D, E) = tf_agc(d, sr)

    # convert back to int16 to save
    y = np.int16(y / np.max(np.abs(y)) * convert_16_bit)
    scipy.io.wavfile.write('speech_agc.wav', sr, y)


## References

D. Ellis (2010), "Time-frequency automatic gain control", web resource, available: [http://labrosa.ee.columbia.edu/matlab/tf_agc/][ref1]



[ref1]: http://labrosa.ee.columbia.edu/matlab/tf_agc/