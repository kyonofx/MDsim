from pathlib import Path
import copy
import logging
import os
import yaml
import torch
from tqdm import tqdm
from torch_geometric.data import Data

from ase import Atoms, units
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator as sp
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import Trajectory

from mdsim.common.registry import registry
from mdsim.common.utils import setup_imports, setup_logging
from mdsim.datasets import data_list_collater

def atoms_to_batch(atoms):
    atomic_numbers = torch.Tensor(atoms.get_atomic_numbers())
    positions = torch.Tensor(atoms.get_positions())
    cell = torch.Tensor(atoms.get_cell()).view(1, 3, 3)
    natoms = positions.shape[0]

    return Data(
        cell=cell,
        pos=positions,
        atomic_numbers=atomic_numbers,
        natoms=natoms,
    )

def data_to_atoms(data):
    numbers = data.atomic_numbers
    positions = data.pos
    cell = data.cell.squeeze()
    atoms = Atoms(numbers=numbers, 
                  positions=positions.cpu().detach().numpy(), 
                  cell=cell.cpu().detach().numpy(),
                  pbc=[True, True, True])
    return atoms

def batch_to_atoms(batch):
    n_systems = batch.natoms.shape[0]
    natoms = batch.natoms.tolist()
    numbers = torch.split(batch.atomic_numbers, natoms)
    forces = torch.split(batch.force, natoms)
    positions = torch.split(batch.pos, natoms)
    # tags = torch.split(batch.tags, natoms)
    cells = batch.cell
    if batch.y is not None:
        energies = batch.y.tolist()
    else:
        energies = [None] * n_systems

    atoms_objects = []
    for idx in range(n_systems):
        atoms = Atoms(
            numbers=numbers[idx].tolist(),
            positions=positions[idx].cpu().detach().numpy(),
            cell=cells[idx].cpu().detach().numpy(),
            pbc=[True, True, True],
        )
        calc = sp(
            atoms=atoms,
            energy=energies[idx],
            forces=forces[idx].cpu().detach().numpy(),
        )
        atoms.set_calculator(calc)
        atoms_objects.append(atoms)

    return atoms_objects


class OCPCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, config_yml=None, checkpoint=None, 
                 test_data_src=None, energy_units_to_eV=1.):
        """
        OCP-ASE Calculator. The default unit for energy is eV.

        Args:
            config_yml (str):
                Path to yaml config or could be a dictionary.
            checkpoint (str):
                Path to trained checkpoint.
        """
        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint is not None

        if config_yml is not None:
            if isinstance(config_yml, str):
                config = yaml.safe_load(open(config_yml, "r"))

                if "includes" in config:
                    for include in config["includes"]:
                        # Change the path based on absolute path of config_yml
                        path = os.path.join(
                            config_yml.split("configs")[0], include
                        )
                        include_config = yaml.safe_load(open(path, "r"))
                        config.update(include_config)
            else:
                config = config_yml
        else:
            # Loads the config from the checkpoint directly
            config = torch.load(checkpoint, map_location=torch.device("cpu"))[
                "config"
            ]

            config["trainer"] = "forces"
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        # Save config so obj can be transported over network (pkl)
        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = checkpoint
        
        for cfg in config['dataset']:
            cfg['src'] = test_data_src
        config['dataset'].append({'src': test_data_src, 'name': 'test'})
        
        # config['dataset'].append(
        #     {'src': '/'.join(self.config['dataset'][1]['src'].split('/')[:-1] + ['test'])})

        self.trainer = registry.get_trainer_class(
            config.get("trainer", "energy")
        )(
            task=config["task"],
            model=config["model"],
            dataset=config["dataset"],
            optimizer=config["optim"],
            identifier=config["identifier"],
            timestamp_id=config.get("timestamp_id", None),
            run_dir=config.get("run_dir", None),
            is_debug=config.get("is_debug", False),
            print_every=config.get("print_every", 100),
            seed=config.get("seed", 0),
            logger=config.get("logger", "wandb"),
            local_rank=config["local_rank"],
            amp=config.get("amp", False),
            cpu=config.get("cpu", False),
            slurm=config.get("slurm", {}),
            no_energy=config.get("no_energy", False),
            simulate=True
        )
        
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)
            
        self.energy_units_to_eV = energy_units_to_eV

    def load_checkpoint(self, checkpoint_path):
        """
        Load existing trained model

        Args:
            checkpoint_path: string
                Path to trained model
        """
        try:
            self.trainer.load_checkpoint(checkpoint_path)
        except NotImplementedError:
            logging.warning("Unable to load checkpoint!")

    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        data_object = atoms_to_batch(atoms)
        batch = data_list_collater([data_object], otf_graph=True)

        predictions = self.trainer.predict(
            batch, per_image=False, disable_tqdm=True
        )
        
        self.results["energy"] = predictions["energy"].item() * self.energy_units_to_eV
        self.results["forces"] = predictions["forces"].cpu().numpy() * self.energy_units_to_eV
            

class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 start_time=0,
                 verbose=True,
                 **kwargs):
        if start_time == 0:
            header = True
        else:
            header = False
        super().__init__(header=header, *args, **kwargs)
        """
        Logger uses ps units.
        """
        self.start_time = start_time
        self.verbose = verbose
        if verbose:
            print(self.hdr)
        self.natoms = self.atoms.get_number_of_atoms()

    def __call__(self):
        if self.start_time > 0 and self.dyn.get_time() == 0:
            return
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000*units.fs) + self.start_time
            dat = (t,)
        else:
            dat = ()
        dat += (epot+ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)

class Simulator:
    def __init__(self, 
                 atoms, 
                 integrator,
                 T_init,
                 start_time=0,
                 save_dir='./log',
                 restart=False,
                 save_frequency=100,
                 min_temp=0.1,
                 max_temp=100000):
        self.atoms = atoms
        self.integrator = integrator
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.natoms = self.atoms.get_number_of_atoms()

        # intialize system momentum 
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, T_init * units.kB)
        
        # attach trajectory dump 
        self.traj = Trajectory(self.save_dir / 'atoms.traj', 'a', self.atoms)
        self.integrator.attach(self.traj.write, interval=save_frequency)
        
        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, self.atoms, 
                                        self.save_dir / 'thermo.log', 
                                        start_time=start_time, mode='a'), 
                               interval=save_frequency)
        
    def run(self, steps):
        early_stop = False
        step = 0
        for step in tqdm(range(steps)):
            self.integrator.run(1)
            
            ekin = self.atoms.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB * self.natoms)
            if temp < self.min_temp or temp > self.max_temp:
                print(f'Temprature {temp:.2f} is out of range: \
                        [{self.min_temp:.2f}, {self.max_temp:.2f}]. \
                        Early stopping the simulation.')
                early_stop = True
                break
            
        self.traj.close()
        return early_stop, (step+1)