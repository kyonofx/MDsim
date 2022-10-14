"""Test trained DeePMD model."""
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Dict, Optional, Tuple

import numpy as np
from deepmd import DeepPotential
from deepmd.common import expand_sys_str
from deepmd.utils import random as dp_random
from deepmd.utils.data import DeepmdData
from deepmd.utils.weight_avg import weighted_average

if TYPE_CHECKING:
    from deepmd.infer import DeepPot
    from deepmd.infer.deep_tensor import DeepTensor

from deepmd.env import default_tf_session_config, tf
from tensorflow.python.platform import gfile
from tensorflow.python.framework import tensor_util

log = logging.getLogger(__name__)

def get_param_count(graph_pb_path):
    with tf.compat.v1.Session(config=default_tf_session_config) as sess:
      with gfile.FastGFile(graph_pb_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')
        graph_nodes = [n for n in graph_def.node]
        wts = [n for n in graph_nodes if n.op == 'Const']

    shapes = []
    for n in wts:
        shapes.append(tensor_util.MakeNdarray(n.attr['value'].tensor).shape)
        
    return sum([np.prod(x) for x in shapes])

def dp_test(
    *,
    model: str,
    system: str,
    numb_test: int,
    batch_size=1000,
    set_prefix='set',
    rand_seed=None,
    shuffle_test=False,
    detail_file=None,
    atomic=False,
    **kwargs,
):
    """Test model predictions.

    Parameters
    ----------
    model : str
        path where model is stored
    system : str
        system directory
    set_prefix : str
        string prefix of set
    numb_test : int
        munber of tests to do
    rand_seed : Optional[int]
        seed for random generator
    shuffle_test : bool
        whether to shuffle tests
    detail_file : Optional[str]
        file where test details will be output
    atomic : bool
        whether per atom quantities should be computed

    Raises
    ------
    RuntimeError
        if no valid system was found
    """
    all_sys = expand_sys_str(system)
    if len(all_sys) == 0:
        raise RuntimeError("Did not find valid system")

    # init random seed
    if rand_seed is not None:
        dp_random.seed(rand_seed % (2 ** 32))

    # init model
    dp = DeepPotential(model)
    
    for cc, system in enumerate(all_sys):
        log.info("# ---------------output of dp test--------------- ")
        log.info(f"# testing system : {system}")

        # create data class
        tmap = dp.get_type_map() if dp.model_type == "ener" else None
        data = DeepmdData(system, set_prefix, shuffle_test=shuffle_test, type_map=tmap)

        err = test_ener(
            dp,
            data,
            system,
            numb_test,
            detail_file,
            atomic,
            batch_size=batch_size,
            append_detail=(cc != 0),
        )
        log.info("# ----------------------------------------------- ")
        
    return err

def rmse(diff: np.ndarray) -> np.ndarray:
    """Calculate average root mean square error.

    Parameters
    ----------
    diff: np.ndarray
        difference

    Returns
    -------
    np.ndarray
        array with normalized difference
    """
    return np.sqrt(np.average(diff * diff))


def save_txt_file(
    fname: Path, data: np.ndarray, header: str = "", append: bool = False
):
    """Save numpy array to test file.

    Parameters
    ----------
    fname : str
        filename
    data : np.ndarray
        data to save to disk
    header : str, optional
        header string to use in file, by default ""
    append : bool, optional
        if true file will be appended insted of overwriting, by default False
    """
    flags = "ab" if append else "w"
    with fname.open(flags) as fp:
        np.savetxt(fp, data, header=header)


def test_ener(
    dp: "DeepPot",
    data: DeepmdData,
    system: str,
    numb_test: int,
    detail_file: Optional[str],
    has_atom_ener: bool,
    batch_size: int,
    append_detail: bool = False,
) -> Tuple[List[np.ndarray], List[int]]:
    """Test energy type model.

    Parameters
    ----------
    dp : DeepPot
        instance of deep potential
    data: DeepmdData
        data container object
    system : str
        system directory
    numb_test : int
        munber of tests to do
    detail_file : Optional[str]
        file where test details will be output
    has_atom_ener : bool
        whether per atom quantities should be computed
    append_detail : bool, optional
        if true append output detail file, by default False

    Returns
    -------
    Tuple[List[np.ndarray], List[int]]
        arrays with results and their shapes
    """
    data.add("energy", 1, atomic=False, must=False, high_prec=True)
    data.add("force", 3, atomic=True, must=False, high_prec=False)
    data.add("virial", 9, atomic=False, must=False, high_prec=False)
    if dp.has_efield:
        data.add("efield", 3, atomic=True, must=True, high_prec=False)
    if has_atom_ener:
        data.add("atom_ener", 1, atomic=True, must=True, high_prec=False)
    if dp.get_dim_fparam() > 0:
        data.add(
            "fparam", dp.get_dim_fparam(), atomic=False, must=True, high_prec=False
        )
    if dp.get_dim_aparam() > 0:
        data.add("aparam", dp.get_dim_aparam(), atomic=True, must=True, high_prec=False)

    test_data = data.get_test()
    natoms = len(test_data["type"][0])
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    if dp.has_efield:
        efield = test_data["efield"][:numb_test].reshape([numb_test, -1])
    else:
        efield = None
    if not data.pbc:
        box = None
    atype = test_data["type"][0]
    if dp.get_dim_fparam() > 0:
        fparam = test_data["fparam"][:numb_test]
    else:
        fparam = None
    if dp.get_dim_aparam() > 0:
        aparam = test_data["aparam"][:numb_test]
    else:
        aparam = None
    
    all_ener = []
    all_force = []
    n_batches = int(np.ceil(numb_test / batch_size))
    splits = np.array_split(np.arange(numb_test), n_batches)
    for i in range(n_batches):
        ret = dp.eval(
            coord[splits[i]],
            box[splits[i]],
            atype,
            fparam=fparam,
            aparam=aparam,
            atomic=has_atom_ener,
            efield=efield,
        )
        energy = ret[0]
        force = ret[1]
        all_ener.append(energy)
        all_force.append(force)
    all_ener = np.concatenate(all_ener)
    all_force = np.concatenate(all_force)
    energy = all_ener.reshape([numb_test, 1])
    force = all_force.reshape([numb_test, -1])

    mae_e = (np.abs(energy - test_data["energy"][:numb_test].reshape([-1, 1]))).mean()
    mae_f = (np.abs((force - test_data["force"][:numb_test]))).mean()
    mae_ea = mae_e / natoms
    
    return {
        "mae_e": mae_e,
        "mae_ea": mae_ea,
        "mae_f": mae_f,
    }

def run_test(dp: "DeepTensor", test_data: dict, numb_test: int):
    """Run tests.

    Parameters
    ----------
    dp : DeepTensor
        instance of deep potential
    test_data : dict
        dictionary with test data
    numb_test : int
        munber of tests to do

    Returns
    -------
    [type]
        [description]
    """
    nframes = test_data["box"].shape[0]
    numb_test = min(nframes, numb_test)

    coord = test_data["coord"][:numb_test].reshape([numb_test, -1])
    box = test_data["box"][:numb_test]
    atype = test_data["type"][0]
    prediction = dp.eval(coord, box, atype)

    return prediction.reshape([numb_test, -1]), numb_test, atype