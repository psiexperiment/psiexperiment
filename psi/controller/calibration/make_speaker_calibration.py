import argparse
import tables as tb
import pylab as pl

from psi.controller.calibration import util


if __name__ == '__main__':
    desc = 'Convert to speaker calibration'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('filename', type=str, help='filename')
    parser.add_argument('smoothing', type=float)
    args = parser.parse_args()

    with tb.open_file(args.filename, 'a') as fh:
        fs = fh.root._v_attrs['fs']
        output_gain = fh.root._v_attrs['output_gain']
        mic_sens = fh.root._v_attrs['ref_mic_sens']
        discard = fh.root._v_attrs['discard']
        n = fh.root._v_attrs['n']

        a = fh.root.a.read()
        b = fh.root.b.read()
        ar = fh.root.a_response.read()[discard:, 0]
        br = fh.root.b_response.read()[discard:, 0]

        freq, vrms, phase = util.golay_tf(a, b, ar, br, fs)
        vrms = vrms.mean(axis=0)
        phase = phase.mean(axis=0)
        _, sig_vrms, _ = util.golay_tf(a, b, a, b, fs)

        # actual output of speaker
        spl = util.db(vrms)-util.db(mic_sens)-util.db(20e-6)

        # correct speaker output so it represents gain of 0 and Vrms=1
        norm_spl = spl-output_gain-util.db(sig_vrms)

        # Calculate sensitivity of speaker as dB (Vrms/Pa)
        sens = -norm_spl-util.db(20e-6)
        sens_smoothed = util.freq_smooth(freq, sens, args.smoothing)

        if 'smoothed_sensitivity' in fh.root:
            fh.root.smoothed_sensitivity._f_remove()
        if 'smoothed_phase' in fh.root:
            fh.root.smoothed_phase._f_remove()

        node = fh.create_array(fh.root, 'smoothed_sensitivity', sens_smoothed)
        node._v_attrs['smoothing'] = args.smoothing
        node = fh.create_array(fh.root, 'smoothed_phase', sens)
        node._v_attrs['smoothing'] = args.smoothing

        pl.semilogx(freq, sens, 'k-')
        pl.semilogx(freq, sens_smoothed, 'r-', lw=2)
        pl.show()
