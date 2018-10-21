import h5py
import matplotlib.pylab as plt
import numpy as np
#matplotlib.rcParams['axes.color_cycle'] = viridis

def plot(hdfname):
    f = h5py.File(hdfname, 'r')

    with h5py.File(hdfname, 'r') as f:
        for groupname in f.keys():
            node = f[groupname]
            info_dict = dict(zip([i.decode("utf-8") for i in node['parameters']['parameters']], node['parameters']['data']))
            plt.scatter(node['lc'][0], node['lc'][1], color='black', marker='.', label=str(info_dict['flag']))
            #plt.scatter(node['lc'][0], np.ones(201)*(1-info_dict['rprs']**2.0), marker='.', color='g', alpha=0.2, label='depth')
            plt.scatter(node['lc'][0], np.ones(201)*(1-0.7*info_dict['std_nobining']), marker='.', color='orange', alpha=0.5, label='0.5sigma(before binning)')
            plt.scatter(node['lc'][0], np.ones(201)*(1-3.0*info_dict['std_nobining']), marker='.', color='green', alpha=0.5, label='3sigma(before binning)')
            plt.scatter(node['lc'][0], np.ones(201)*(1-3.0*info_dict['std_bining']), marker='.', color='steelblue', alpha=0.5, label='3sigma(after binning)')


            plt.legend(loc='upper left')

            plt.show()
            print(info_dict['rprs'],groupname)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='plot sim result')
    parser.add_argument("fname", help='name of hdfname', type=str)
    #parser.add_argument("group", help='which lc', type=int)
    args = parser.parse_args()
    plot(args.fname)
