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
    url = 'https://zenodo.org/record/7196767/files/alanine_dipeptide.npy?download=1'
    request.urlretrieve(url, os.path.join(data_path, 'alanine_dipeptide.npy'))
    
def write_to_lmdb(data_path, db_path, time_split):
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=False,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )
    
    data_file = (Path(data_path) / 'alanine_dipeptide.npy')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not data_file.is_file():
        download(data_path)
        
    n_points = 50000
    all_data = np.load(data_file, allow_pickle=True).item()
    all_data['force'] = all_data['force'] / EV_TO_KCAL_MOL
    force = all_data['force']
    
    if time_split:
        test = np.arange(n_points-10000, n_points)
    else:
        train_val_pool, test = train_test_split(np.arange(n_points), train_size=n_points-10000, 
                                        test_size=10000, random_state=123)
    for dataset_size, train_size, val_size in zip(['40k'], [38000], [2000]):
        print(f'processing dataset with size {dataset_size}.')
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
            'e_mean': 0,
            'e_std': 1,
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
        
            for i, idx in enumerate(tqdm((ranges[spidx]))):
                data = {k: v[idx] if (v.shape[0] == 50001) else v for k, v in all_data.items()}
                natoms = np.array([data['pos'].shape[0]] * 1, dtype=np.int64)
                data = a2g.convert(natoms, data['pos'], data['atomic_number'], 
                                data['lengths'][None, :], data['angles'][None, :], forces=data['force'])
                txn = db.begin(write=True)
                txn.put(f"{i}".encode("ascii"), pickle.dumps(data, protocol=-1))
                txn.commit()

            # Save count of objects in lmdb.
            txn = db.begin(write=True)
            txn.put("length".encode("ascii"), pickle.dumps(i, protocol=-1))
            txn.commit()

            db.sync()
            db.close()
                        
            # for nequip. turn energy loss == 0.
            data = all_data
            data['pbc'] = np.array([True]*3)
            data = {k: v[ranges[spidx]] if v.shape[0] == 50001 else v for k, v in data.items()}
            data['energy'] = np.zeros(len(ranges[spidx]))[:, None]
            data['force'] = data['force']
            data['lattices'] = data['lengths'][:, None] * np.eye(3)
            np.savez(save_path / 'nequip_npz', **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./DATAPATH/ala')
    parser.add_argument("--db_path", type=str, default='./DATAPATH/ala')
    parser.add_argument("--time_split", action="store_true", help='split data by time order')
    args = parser.parse_args()
    write_to_lmdb(args.data_path, args.db_path, args.time_split)
