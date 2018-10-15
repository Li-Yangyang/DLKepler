import pandas as pd
import kplr
import kepio
import time
import urllib
cata_path = '../catalog/keplerstellar1'
catalog= pd.read_csv(cata_path)
allid = kepio.get_id(catalog)
i = 0
s = time.time()
while i < len(allid):
    try:
        client = kplr.API('/media/yinanzhao/Backup_7/kepdata')
        lcs = client.light_curves(allid[i], fetch = True)
        i = i + 1
    except (TimeoutError, urllib.error.URLError) as e:
        print(e, allid[i]) 
        if  str(e)=='HTTP Error 404: Not Found':
            i = i + 1
        else: 
            pass

e = time.time()
print(e-s)
