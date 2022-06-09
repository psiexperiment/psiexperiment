from pathlib import Path

import numpy as np
import pandas as pd


def norm(x, shrink=1):
    result = (x - x.min()) / (x.max() - x.min())
    offset = (1 - shrink) / 2
    return result * shrink + offset


def get_normalized_filename(n_channels, include_exg=True):
    if include_exg:
        filename = f'mapped_electrode_coords_{n_channels}_exg.csv'
    else:
        filename = f'mapped_electrode_coords_{n_channels}.csv'
    return Path(__file__).parent/ filename


def load_normalized_coords(n_channels, include_exg=True):
    filename = get_normalized_filename(n_channels, include_exg)
    return pd.read_csv(filename).set_index('electrode')


def load_coords(n_channels, include_exg=True, radius=1, shrink=1):
    path = Path(__file__).parent / f'electrode_coords_{n_channels}.csv'
    coords = _read_coords(path, radius, shrink)
    coords['type'] = 'electrode'
    if len(coords) != n_channels:
        raise ValueError('EEG coordinates malformed')
    if include_exg:
        path = Path(__file__).parent / 'exg_coords.csv'
        exg_coords = pd.read_csv(path, index_col=0)
        if len(exg_coords) != 8:
            raise ValueError('EXG coordinates malformed')
        exg_coords['type'] = 'EXG'
        coords = pd.concat((coords, exg_coords))

    coords['index'] = range(len(coords))
    return coords


def convert_coords(n_channels, include_exg=True, radius=1, shrink=0.8):
    coords = load_coords(n_channels, include_exg, radius, shrink)
    filename = get_normalized_filename(n_channels, include_exg)
    cols = ['index', 'electrode', 'x_norm', 'y_norm', 'type']
    coords.reset_index()[cols].round(2).to_csv(filename, index=False)


def _read_coords(path, radius, shrink):
    coords = pd.read_csv(path, index_col=0)

    coords = np.deg2rad(coords)
    coords['x'] = coords.eval('@radius * sin(inclination) * cos(azimuth)')
    coords['y'] = coords.eval('@radius * sin(inclination) * sin(azimuth)')
    coords['z'] = coords.eval('@radius * cos(inclination)')

    radius_proj = coords.eval('abs(@radius / cos(inclination))')
    radius_map = {v: i+4 for i, v in enumerate(sorted(set(radius_proj)))}
    coords['radius_proj'] = radius_proj
    coords['radius_proj_index'] = coords['radius_proj'].map(radius_map)

    coords['x_proj'] = coords.eval('log(radius_proj_index) * sin(inclination) * cos(azimuth)')
    coords['y_proj'] = coords.eval('log(radius_proj_index) * sin(inclination) * sin(azimuth)')

    coords['x_norm'] = norm(coords['x_proj'], shrink)
    coords['y_norm'] = norm(coords['y_proj'], shrink)
    return coords


if __name__ == '__main__':
    convert_coords(32, True, 1)
    convert_coords(64, True, 1)
    convert_coords(32, False, 1)
    convert_coords(64, False, 1)
