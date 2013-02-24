if __name__ == '__main__':
    import scipy.io.wavfile
    import numpy as np
    from agc import tf_agc

    # read audiofile
    sr, d = scipy.io.wavfile.read('speech.wav')

    # convert from int16 to float (-1,1) range
    convert_16_bit = float(2 ** 15)
    d = d / (convert_16_bit + 1.0)

    # apply AGC
    (y, D, E) = tf_agc(d, sr, plot=True)

    # convert back to int16 to save
    y = np.int16(y / np.max(np.abs(y)) * convert_16_bit)
    scipy.io.wavfile.write('speech_agc.wav', sr, y)
