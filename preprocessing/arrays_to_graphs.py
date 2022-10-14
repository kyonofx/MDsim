import torch
from torch_geometric.data import Data
import mdsim.common.utils as utils


class AtomsToGraphs:
    """A class to help convert periodic atomic structures to graphs.

    The AtomsToGraphs class takes in periodic atomic structures in form of ASE atoms objects and converts
    them into graph representations for use in PyTorch. The primary purpose of this class is to determine the
    nearest neighbors within some radius around each individual atom, taking into account PBC, and set the
    pair index and distance between atom pairs appropriately. Lastly, atomic properties and the graph information
    are put into a PyTorch geometric data object for use with PyTorch.

    Args:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstroms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.

    Attributes:
        max_neigh (int): Maximum number of neighbors to consider.
        radius (int or float): Cutoff radius in Angstoms to search for neighbors.
        r_energy (bool): Return the energy with other properties. Default is False, so the energy will not be returned.
        r_forces (bool): Return the forces with other properties. Default is False, so the forces will not be returned.
        r_distances (bool): Return the distances with other properties.
        Default is False, so the distances will not be returned.

    """

    def __init__(
        self,
        max_neigh=200,
        radius=6,
        r_energy=False,
        r_forces=False,
        r_distances=False,
        r_edges=True,
        device='cpu'
    ):
        self.max_neigh = max_neigh
        self.radius = radius
        self.r_energy = r_energy
        self.r_forces = r_forces
        self.r_distances = r_distances
        self.r_edges = r_edges
        self.device = device

    def convert(
        self,
        natoms,
        positions,
        atomic_numbers,
        lengths=None,
        angles=None,
        energy=None,
        forces=None,
        cell=None,
    ):
        """Convert a batch of atomic stucture to a batch of graphs.

        Args:
            natoms: (B), sum(natoms) == N
            positions: (B*N, 3)
            atomic_numbers: (B*N)
            lengths: (B, 3) lattice lengths [lx, ly, lz]
            angles: (B, 3) lattice angles [ax, ay, az] 
            forces: (B*N, 3)
            energy: (B)

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with edge_index, positions, atomic_numbers,
            and optionally, energy, forces, and distances.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        natoms = torch.from_numpy(natoms).to(self.device).long()
        positions = torch.from_numpy(positions).to(self.device).float()
        atomic_numbers = torch.from_numpy(atomic_numbers).to(self.device).long()
        if cell is None:
            lengths = torch.from_numpy(lengths).to(self.device).float()
            angles = torch.from_numpy(angles).to(self.device).float()
            cells = utils.lattice_params_to_matrix_torch(lengths, angles).float()
        else:
            cells = torch.from_numpy(cell).to(self.device).float()
        
        data = Data(
            cell=cells,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
        )

        # optionally include other properties
        if self.r_edges:
            edge_index, cell_offsets, edge_distances, _ = utils.radius_graph_pbc(
                data, self.radius, self.max_neigh)
            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
        if energy is not None:
            energy = torch.from_numpy(energy).to(self.device).float()
            data.y = energy
        if forces is not None:
            forces = torch.from_numpy(forces).to(self.device).float()
            data.force = forces
        if self.r_distances and self.r_edges:
            data.distances = edge_distances  
        
        fixed_idx = torch.zeros(natoms).float()
        data.fixed = fixed_idx
            
        return data.cpu()