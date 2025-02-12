import glob
import numpy as np
import astropy.units as u
from astropy.io import fits

from measure_extinction.extdata import ExtData


if __name__ == "__main__":

    fpath = "data/gordon24/"

    files = glob.glob(f"{fpath}*.fits")

    for fname in files:
        ifile = fname
        ext = ExtData(ifile)
 
        # just need to rename
        ofile = ifile.replace("gordon24/", "gor24_")
        ext.save(ofile)
