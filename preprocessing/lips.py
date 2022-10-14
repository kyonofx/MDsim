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
from ase.io import read

def download(data_path):
    url = 'https://archive.materialscloud.org/record/file?filename=lips.xyz&record_id=1302'
    request.urlretrieve(url, os.path.join(data_path, 'lips.xyz'))

def write_to_lmdb(data_path, db_path):
    a2g = AtomsToGraphs(
        max_neigh=1000,
        radius=4.,
        r_energy=True,
        r_forces=True,
        r_distances=False,
        r_edges=False,
        device='cpu'
    )
    
    data_file = (Path(data_path) / 'lips.xyz')
    Path(data_path).mkdir(parents=True, exist_ok=True)
    if not data_file.is_file():
        download(data_path)
        
    atoms = read(data_file, index=':', format='extxyz')
    n_points = len(atoms)
    positions, cell, atomic_numbers, energy, forces = [], [], [], [], []
    for i in range(n_points):
        positions.append(atoms[i].get_positions())
        cell.append(atoms[i].get_cell())
        atomic_numbers.append(atoms[i].get_atomic_numbers())
        energy.append(atoms[i].get_potential_energy())
        forces.append(atoms[i].get_forces())
    positions = np.array(positions)
    cell = np.array(cell)[0]
    atomic_numbers = np.array(atomic_numbers)[0]
    energy = np.array(energy)[:, None] 
    forces = np.array(forces)
        
    for dataset_size, train_size, val_size in zip(['20k'], [19000], [1000]):
        print(f'processing dataset with size {dataset_size}.')
        size = train_size + val_size
        train, test = train_test_split(np.arange(n_points), train_size=size, test_size=n_points-size, random_state=123)
        train, val = train_test_split(train, train_size=train_size, test_size=val_size, random_state=123)
        ranges = [train, val, test]
        
        norm_stats = {
            'e_mean': energy[train].mean(),
            'e_std': energy[train].std(),
            'f_mean': forces[train].mean(),
            'f_std': forces[train].std(),
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
                
                natoms = np.array([atomic_numbers.shape[0]] * 1, dtype=np.int64)
                data = a2g.convert(natoms, positions[idx], atomic_numbers, 
                                energy=energy[idx], forces=forces[idx], cell=cell[None, :])
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
            data = {}
            data['pbc'] = np.array([True]*3)
            data['pos'] = positions[ranges[spidx]]
            data['energy'] = energy[ranges[spidx]]
            data['forces'] = forces[ranges[spidx]]
            data['cell'] = cell
            data['atomic_numbers'] = atomic_numbers
            np.savez(save_path / 'nequip_npz', **data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default='./DATAPATH/lips')
    parser.add_argument("--db_path", type=str, default='./DATAPATH/lips')
    args = parser.parse_args()
    write_to_lmdb(args.data_path, args.db_path)