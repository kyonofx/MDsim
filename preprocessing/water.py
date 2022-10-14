import os
import argparse
from pathlib import Path
import pickle

import lmdb
import numpy as np
from tqdm import tqdm
from urllib import request as request

from arrays_to_graphs import AtomsToGraphs
from sklearn.model_selection import train_test_split

from mdsim.common.utils import EV_TO_KCAL_MOL

def download(data_path):
    url = 'https://zenodo.org/record/7196767/files/water.npy?download=1'
    request.urlretrieve(url, os.path.join(data_path, 'water.npy'))
    
def write_to_lmdb(data_path, db_path, time_split):
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=4.,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )
    
    data_file = (Path(data_path) / 'water.npy')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not data_file.is_file():
        download(data_path)
        
    n_points = 100001
    all_data = np.load(data_file, allow_pickle=True).item()
    all_data['energy'] = all_data['energy'] / EV_TO_KCAL_MOL
    all_data['forces'] = all_data['forces'] / EV_TO_KCAL_MOL
    energy = all_data['energy']
    force = all_data['forces']
    
    if time_split:
        test = np.arange(90000, n_points)
    else:
        train_val_pool, test = train_test_split(np.arange(n_points), train_size=n_points-10000, 
                                       test_size=10000, random_state=123)
    
    for dataset_size, train_size, val_size in zip(['1k', '10k', '90k'], [950, 9500, 85500], [50, 500, 4500]):
        print(f'processing dataset with size {dataset_size}.')
        # time split
        if time_split:
            train = np.arange(train_size)
            val = np.arange(train_size, train_size+val_size)
            dataset_size = dataset_size + '_time_split'
        else:
            size = train_size + val_size
            train_val = train_val_pool[:size]
            train, val = train_test_split(train_val, train_size=train_size, test_size=val_size, random_state=123)
        ranges = [train, val, test]
        
        norm_stats = {
            'e_mean': energy[train].mean(),
            'e_std': energy[train].std(),
            'f_mean': force[train].mean(),
            'f_std': force[train].std(),
        }
        save_path = Path(db_path) / dataset_size
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / 'metadata', norm_stats)
    
        for spidx, split in enumerate(['train', 'val', 'test']):
            print(f'processing split {split}.')
            # for OCP
            save_path = Path(db_path) / dataset_size / split
            save_path.mkdir(parents=True, exist_ok=True)
            db = lmdb.open(
                str(save_path / 'data.lmdb'),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )
        
            for i, idx in enumerate(tqdm(ranges[spidx])):
                
                data = {k: v[idx] if (v.shape[0] == 100001) else v for k, v in all_data.items()}
                natoms = np.array([data['wrapped_coords'].shape[0]] * 1, dtype=np.int64)
                data = a2g.convert(natoms, data['wrapped_coords'], data['atom_types'], 
                                data['lengths'][None, :], data['angles'][None, :], 
                                np.array([data['energy']]), data['forces'])
                data.sid = 0
                data.fid = idx
                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()
        
            # Save count of objects in lmdb.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
            txn.commit()

            db.sync()
            db.close()
            
            # for nequip.
            data = all_data
            data['pbc'] = np.array([True]*3)
            data = {k: v[ranges[spidx]] if v.shape[0] == 100001 else v for k, v in data.items()}
            data['energy'] = energy[ranges[spidx]][:, None] / EV_TO_KCAL_MOL
            data['force'] = force[ranges[spidx]] / EV_TO_KCAL_MOL
            data['lattices'] = data['lengths'][:, None] * np.eye(3)
            np.savez(save_path / 'nequip_npz', **data)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./DATAPATH/lips')
    parser.add_argument("--db_path", type=str, default='./DATAPATH/lips')
    parser.add_argument("--time_split", action="store_true", help='split data by time order')
    args = parser.parse_args()
    write_to_lmdb(args.data_path, args.db_path, args.time_split)
