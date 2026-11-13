"""Convert the MATLAB EPS figures to PDF and render PNG previews.

Requires epstopdf (TeX Live) and PyMuPDF. The PDF is generated from the legacy
MATLAB EPS renderer so its dash patterns match the manuscript figures.
No curve data are modified here.
"""
import os
from pathlib import Path
import shutil
import subprocess

import fitz

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / 'figures' / 'r0_style'


def main():
    executable = shutil.which('epstopdf')
    if not executable:
        raise RuntimeError('epstopdf is required; add your TeX Live bin directory to PATH.')
    env = dict(os.environ, LC_ALL='C', LANG='C')
    eps_files = sorted(FIGURES.glob('Fig_A_*.eps'))
    if len(eps_files) != 7:
        raise RuntimeError('Run make_figures_r0 in MATLAB to generate all seven figures first.')
    for eps in eps_files:
        pdf = eps.with_suffix('.pdf')
        subprocess.run([executable, str(eps), '--outfile=' + str(pdf)], check=True, env=env)
        with fitz.open(pdf) as document:
            if len(document) != 1:
                raise RuntimeError(f'Expected one figure page: {pdf}')
            document[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(str(eps.with_suffix('.png')))
        print('Finalized', pdf.name)


if __name__ == '__main__':
    main()
