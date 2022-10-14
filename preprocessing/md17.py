import os
import argparse
from pathlib import Path
import pickle

import lmdb
import numpy as np
from tqdm import tqdm
from urllib import request as request
from sklearn.model_selection import train_test_split

from arrays_to_graphs import AtomsToGraphs
from mdsim.common.utils import EV_TO_KCAL_MOL

MD17_mols = ['aspirin', 'benzene', 'ethanol', 'malonaldehyde', 
             'naphthalene', 'salicylic_acid', 'toluene', 'uracil']

datasets_dict = dict(
        aspirin="aspirin_dft.npz",
        azobenzene="azobenzene_dft.npz",
        benzene="benzene2017_dft.npz",
        ethanol="ethanol_dft.npz",
        malonaldehyde="malonaldehyde_dft.npz",
        naphthalene="naphthalene_dft.npz",
        paracetamol="paracetamol_dft.npz",
        salicylic_acid="salicylic_dft.npz",
        toluene="toluene_dft.npz",
        uracil="uracil_dft.npz")

def download(molecule, data_path):
    url = (
        "http://www.quantum-machine.org/gdml/data/npz/"
        + datasets_dict[molecule]
    )        
    request.urlretrieve(url, os.path.join(data_path, datasets_dict[molecule]))
    print(f'{molecule} downloaded.')

def write_to_lmdb(molecule, data_path, db_path):
    print(f'process MD17 molecule: {molecule}.')
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=6,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )
    
    npzname = datasets_dict[molecule]
    data_file = Path(data_path) / npzname
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not data_file.is_file():
        download(molecule, data_path)
    all_data = np.load(data_file)
    
    n_points = all_data.f.R.shape[0]
    atomic_numbers = all_data.f.z
    atomic_numbers = atomic_numbers.astype(np.int64)
    positions = all_data.f.R
    force = all_data.f.F / EV_TO_KCAL_MOL
    energy = all_data.f.E / EV_TO_KCAL_MOL
    lengths = np.ones(3)[None, :] * 30.
    angles = np.ones(3)[None, :] * 90.
    
    train_val_pool, test = train_test_split(np.arange(n_points), train_size=n_points-10000, 
                                       test_size=10000, random_state=123)
    
    for dataset_size, train_size, val_size in zip(['10k'], [9500], [500]):
        print(f'processing dataset with size {dataset_size}.')
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
        save_path = Path(db_path) / molecule / dataset_size
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / 'metadata', norm_stats)
        
        for spidx, split in enumerate(['train', 'val', 'test']):
            print(f'processing split {split}.')
            save_path = Path(db_path) / molecule / dataset_size / split
            save_path.mkdir(parents=True, exist_ok=True)
            db = lmdb.open(
                str(save_path / 'data.lmdb'),
                map_size=1099511627776 * 2,
                subdir=False,
                meminit=False,
                map_async=True,
            )
            for i, idx in enumerate(tqdm(ranges[spidx])):
                natoms = np.array([positions.shape[1]] * 1, dtype=np.int64)
                data = a2g.convert(natoms, positions[idx], atomic_numbers, 
                                lengths, angles, energy[idx], force[idx])
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

            # nequip
            data = {
                'z': atomic_numbers,
                'E': energy[ranges[spidx]],
                'F': force[ranges[spidx]],
                'R': all_data.f.R[ranges[spidx]]
            }
            np.savez(save_path / 'nequip_npz', **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecule", type=str, default='ethanol')
    parser.add_argument("--data_path", type=str, default='./DATAPATH/md17')
    parser.add_argument("--db_path", type=str, default='./DATAPATH/md17')
    args = parser.parse_args()
    assert args.molecule in MD17_mols, '<molecule> must be one of the 8 molecules in MD17.'
    write_to_lmdb(args.molecule, args.data_path, args.db_path)
