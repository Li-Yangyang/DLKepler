import numpy as np
import kplr
import pandas as pd
import timeit
import cProfile
import pstats
import io
import kepio
from multiprocessing.dummy import Pool
import glob
from astropy.io import fits

#kplr.config.KPLR_ROOT = '/media/yinanzhao/Backup_7/kepdata'

def process(lc):
    with lc.open() as f:
        q = f[0].header['QUARTER']
    return q

def process2(filedir):
    try:
        filename = glob.glob(filedir)[0]
        with fits.open(filename) as f:
            q = f[0].header['QUARTER']
            f.close()
        return q
    except Exception as e:
        return None

def test(kepid):

  client = kplr.API('/media/yinanzhao/Backup_7/kepdata')
  #star = client.star(kepid)
  lcs = client.light_curves(kepid,short_cadence=False)


  with Pool(5) as p:
      q2 = p.map(process, lcs)

  return q2

def test2(kepid):
    filedir = [kepio.pathfinder(kepid, '/scratch/kepler_data/', q) for q in range(18)]
    with Pool(5) as p:
        q2 = p.map(process2, filedir)
    return q2

pr = cProfile.Profile()
pr.enable()
cata_path = '../catalog/cumulative+noise.csv'
catalog= pd.read_csv(cata_path, skiprows=67)
kepid = np.random.choice(catalog['kepid'].values, 10)


for i in range(len(kepid)):
  print(test(kepid[i]))

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('test3.txt', 'w+') as f:
    f.write(s.getvalue())
