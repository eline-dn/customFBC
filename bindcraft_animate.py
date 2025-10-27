####################################
###################### BindCraft Run
####################################
### Import dependencies
import os, re, shutil, time, json, glob
import argparse
import pickle
import warnings
import zipfile
import numpy as np
import pandas as pd
import math, random
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.af.alphafold.common import residue_constants,protein
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.protein import renum_pdb_str

from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, Model, PDBIO, Select, Chain
from Bio.PDB.Selection import unfold_entities

os.environ["SLURM_STEP_NODELIST"] = os.environ["SLURM_NODELIST"]
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=BiopythonWarning)

three_to_one_map = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

# identify interacting residues at the binder interface
def hotspot_residues(trajectory_pdb, binder_chain="B", atom_distance_cutoff=4.0):
    # Parse the PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", trajectory_pdb)

    # Get the specified chain
    binder_atoms = Selection.unfold_entities(structure[0][binder_chain], 'A')
    binder_coords = np.array([atom.coord for atom in binder_atoms])

    # Get atoms and coords for the target chain
    target_atoms = Selection.unfold_entities(structure[0]['A'], 'A')
    target_coords = np.array([atom.coord for atom in target_atoms])

    # Build KD trees for both chains
    binder_tree = cKDTree(binder_coords)
    target_tree = cKDTree(target_coords)

    # Prepare to collect interacting residues
    interacting_residues = {}

    # Query the tree for pairs of atoms within the distance cutoff
    pairs = binder_tree.query_ball_tree(target_tree, atom_distance_cutoff)

    # Process each binder atom's interactions
    for binder_idx, close_indices in enumerate(pairs):
        binder_residue = binder_atoms[binder_idx].get_parent()
        binder_resname = binder_residue.get_resname()

        # Convert three-letter code to single-letter code using the manual dictionary
        if binder_resname in three_to_one_map:
            aa_single_letter = three_to_one_map[binder_resname]
            for close_idx in close_indices:
                target_residue = target_atoms[close_idx].get_parent()
                interacting_residues[binder_residue.id[1]] = aa_single_letter

    return interacting_residues

# calculate secondary structure percentage of design
def calc_ss_percentage(pdb_file, advanced_settings, chain_id="B", atom_distance_cutoff=4.0):
    # Parse the structure
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    model = structure[0]  # Consider only the first model in the structure

    # Calculate DSSP for the model
    dssp = DSSP(model, pdb_file, dssp=advanced_settings["dssp_path"])

    # Prepare to count residues
    ss_counts = defaultdict(int)
    ss_interface_counts = defaultdict(int)
    plddts_interface = []
    plddts_ss = []

    # Get chain and interacting residues once
    chain = model[chain_id]
    interacting_residues = set(hotspot_residues(pdb_file, chain_id, atom_distance_cutoff).keys())

    for residue in chain:
        residue_id = residue.id[1]
        if (chain_id, residue_id) in dssp:
            ss = dssp[(chain_id, residue_id)][2]  # Get the secondary structure
            ss_type = 'loop'
            if ss in ['H', 'G', 'I']:
                ss_type = 'helix'
            elif ss == 'E':
                ss_type = 'sheet'

            ss_counts[ss_type] += 1

            if ss_type != 'loop':
                # calculate secondary structure normalised pLDDT
                avg_plddt_ss = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_ss.append(avg_plddt_ss)

            if residue_id in interacting_residues:
                ss_interface_counts[ss_type] += 1

                # calculate interface pLDDT
                avg_plddt_residue = sum(atom.bfactor for atom in residue) / len(residue)
                plddts_interface.append(avg_plddt_residue)

    # Calculate percentages
    total_residues = sum(ss_counts.values())
    total_interface_residues = sum(ss_interface_counts.values())

    percentages = calculate_percentages(total_residues, ss_counts['helix'], ss_counts['sheet'])
    interface_percentages = calculate_percentages(total_interface_residues, ss_interface_counts['helix'], ss_interface_counts['sheet'])

    i_plddt = round(sum(plddts_interface) / len(plddts_interface) / 100, 2) if plddts_interface else 0
    ss_plddt = round(sum(plddts_ss) / len(plddts_ss) / 100, 2) if plddts_ss else 0

    return (*percentages, *interface_percentages, i_plddt, ss_plddt)

def calculate_percentages(total, helix, sheet):
    helix_percentage = round((helix / total) * 100,2) if total > 0 else 0
    sheet_percentage = round((sheet / total) * 100,2) if total > 0 else 0
    loop_percentage = round(((total - helix - sheet) / total) * 100,2) if total > 0 else 0

    return helix_percentage, sheet_percentage, loop_percentage

def calculate_clash_score(pdb_file, threshold=2.4, only_ca=False):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    atoms = []
    atom_info = []  # Detailed atom info for debugging and processing

    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.element == 'H':  # Skip hydrogen atoms
                        continue
                    if only_ca and atom.get_name() != 'CA':
                        continue
                    atoms.append(atom.coord)
                    atom_info.append((chain.id, residue.id[1], atom.get_name(), atom.coord))

    tree = cKDTree(atoms)
    pairs = tree.query_pairs(threshold)

    valid_pairs = set()
    for (i, j) in pairs:
        chain_i, res_i, name_i, coord_i = atom_info[i]
        chain_j, res_j, name_j, coord_j = atom_info[j]

        # Exclude clashes within the same residue
        if chain_i == chain_j and res_i == res_j:
            continue

        # Exclude directly sequential residues in the same chain for all atoms
        if chain_i == chain_j and abs(res_i - res_j) == 1:
            continue

        # If calculating sidechain clashes, only consider clashes between different chains
        if not only_ca and chain_i == chain_j:
            continue

        valid_pairs.add((i, j))

    return len(valid_pairs)


def mutate_to_glycine(pdb_file, output_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    for model in structure:
        for chain in model:
            if chain.id == 'B':
                for residue in chain:
                    if Polypeptide.is_aa(residue):
                        # Mutate the residue to glycine (set name to GLY)
                        residue.resname = 'GLY'
                        
                        # Get a list of sidechain atoms to remove (keep only N, CA, C, O)
                        atoms_to_remove = [
                            atom for atom in residue if atom.id not in {'N', 'CA', 'C', 'O'}
                        ]
                        
                        # Remove the sidechain atoms
                        for atom in atoms_to_remove:
                            residue.detach_child(atom.id)

    # Save the mutated structure to an output PDB file
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_file)

def delete_files_with_design_name(design_name, design_paths):
    # Get the folder path from design_paths dictionary
    folder_path = design_paths["Trajectory/Frames"]

    # Construct the search pattern to match files starting with design_name
    search_pattern = os.path.join(folder_path, f"{design_name}*")

    # Use glob to find and delete all matching files
    for file_path in glob.glob(search_pattern):
        try:
            os.remove(file_path)
            #print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def process_num_pos(setting, length):
    if setting in (None, False, ""):
        return float("inf")

    if setting is True:
        return math.ceil(length * 0.2)

    if isinstance(setting, int):
        return setting

    if isinstance(setting, float):
        if math.isfinite(setting):
            return math.ceil(length * setting)
        else:
            return float("inf")

    raise ValueError(f"Advanced setting has incorrect value: {setting!r}")

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths, save_trajectory=True):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    if save_trajectory:
        def save_traj_function(self):
            # Save the current PDB
            trajectory_pdb_path = os.path.join(design_paths["Trajectory/Frames"], f"{design_name}_{self._k}.pdb")
            self.save_current_pdb(trajectory_pdb_path)

            # Mutate the saved PDB to glycine and save as a separate file
            mutated_pdb_path = os.path.join(design_paths["Trajectory/Frames"], f"{design_name}_gly_{self._k}.pdb")
            mutate_to_glycine(trajectory_pdb_path, mutated_pdb_path)
            os.remove(trajectory_pdb_path)

        # Use lambda to call save_traj_function with self context
        save_traj = lambda self: save_traj_function(self)

       # save_traj = lambda self: self.save_current_pdb(os.path.join(design_paths["Trajectory/Frames"], design_name+"_"+str(self._k)+".pdb"))
    else:
        save_traj = None

    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss', post_design_callback=save_traj)

    # sanity check for hotspots
    if target_hotspot_residues == "":
        target_hotspot_residues = None

    af_model.prep_inputs(pdb_filename=starting_pdb, chain=chain, binder_len=length, hotspot=target_hotspot_residues, seed=seed, rm_aa=advanced_settings["omit_AAs"],
                        rm_target_seq=advanced_settings["rm_template_seq_design"])

    ### Update weights based on specified settings
    af_model.opt["weights"].update({"pae":advanced_settings["weights_pae_intra"],
                                    "plddt":advanced_settings["weights_plddt"],
                                    "i_pae":advanced_settings["weights_pae_inter"],
                                    "con":advanced_settings["weights_con_intra"],
                                    "i_con":advanced_settings["weights_con_inter"],
                                    })

    intra_contact_num_pos = process_num_pos(advanced_settings["intra_contact_num_pos"], length)
    inter_contact_num_pos = process_num_pos(advanced_settings["inter_contact_num_pos"], length)

    # redefine intramolecular contacts (con) and intermolecular contacts (i_con) definitions
    af_model.opt["con"].update({"num":advanced_settings["intra_contact_number"], "cutoff":advanced_settings["intra_contact_distance"],
                                "binary":False, "num_pos": intra_contact_num_pos, "seqsep":9})
    af_model.opt["i_con"].update({"num":advanced_settings["inter_contact_number"], "cutoff":advanced_settings["inter_contact_distance"],
                                "binary":False, "num_pos":inter_contact_num_pos})

    ### additional loss functions
    if advanced_settings["use_rg_loss"]:
        # radius of gyration loss
        add_rg_loss(af_model, advanced_settings["weights_rg"])

    if advanced_settings["use_i_ptm_loss"]:
        # interface pTM loss
        add_i_ptm_loss(af_model, advanced_settings["weights_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_loss"])

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    # initial logits to prescreen trajectory
    print("Stage 1: Test Logits")
    af_model.design_logits(iters=50, e_soft=0.9, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

    # determine pLDDT of best iteration according to lowest 'loss' value
    initial_plddt = get_best_plddt(af_model, length)
    
    # if best iteration has high enough confidence then continue
    if initial_plddt > 0.65:
        print("Initial trajectory pLDDT good, continuing: "+str(initial_plddt))
        if advanced_settings["optimise_beta"]:
            # temporarily dump model to assess secondary structure
            af_model.save_pdb(model_pdb_path)
            _, beta, *_ = calc_ss_percentage(model_pdb_path, advanced_settings, 'B')
            os.remove(model_pdb_path)

            # if beta sheeted trajectory is detected then choose to optimise
            if float(beta) > 15:
                advanced_settings["soft_iterations"] = advanced_settings["soft_iterations"] + advanced_settings["optimise_beta_extra_soft"]
                advanced_settings["temporary_iterations"] = advanced_settings["temporary_iterations"] + advanced_settings["optimise_beta_extra_temp"]
                af_model.set_opt(num_recycles=advanced_settings["optimise_beta_recycles_design"])
                print("Beta sheeted trajectory detected, optimising settings")

        # how many logit iterations left
        logits_iter = advanced_settings["soft_iterations"] - 50
        if logits_iter > 0:
            print("Stage 1: Additional Logits Optimisation")
            af_model.clear_best()
            af_model.design_logits(iters=logits_iter, e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
                                ramp_recycles=False, save_best=True)
            af_model._tmp["seq_logits"] = af_model.aux["seq"]["logits"]
            logit_plddt = get_best_plddt(af_model, length)
            print("Optimised logit trajectory pLDDT: "+str(logit_plddt))
        else:
            logit_plddt = initial_plddt

        # perform softmax trajectory design
        if advanced_settings["temporary_iterations"] > 0:
            print("Stage 2: Softmax Optimisation")
            af_model.clear_best()
            af_model.design_soft(advanced_settings["temporary_iterations"], e_temp=1e-2, models=design_models, num_models=1,
                                sample_models=advanced_settings["sample_models"], ramp_recycles=False, save_best=True)
            softmax_plddt = get_best_plddt(af_model, length)
        else:
            softmax_plddt = logit_plddt

        # perform one hot encoding
        if softmax_plddt > 0.65:
            print("Softmax trajectory pLDDT good, continuing: "+str(softmax_plddt))
            if advanced_settings["hard_iterations"] > 0:
                af_model.clear_best()
                print("Stage 3: One-hot Optimisation")
                af_model.design_hard(advanced_settings["hard_iterations"], temp=1e-2, models=design_models, num_models=1,
                                sample_models=advanced_settings["sample_models"], dropout=False, ramp_recycles=False, save_best=True)
                onehot_plddt = get_best_plddt(af_model, length)

            if onehot_plddt > 0.65:
                # perform greedy mutation optimisation
                print("One-hot trajectory pLDDT good, continuing: "+str(onehot_plddt))
                if advanced_settings["greedy_iterations"] > 0:
                    print("Stage 4: PSSM Semigreedy Optimisation")
                    af_model.clear_best()
                    af_model.design_pssm_semigreedy(soft_iters=0, hard_iters=advanced_settings["greedy_iterations"], tries=greedy_tries, models=design_models, 
                                                    num_models=1, sample_models=advanced_settings["sample_models"], ramp_models=False, save_best=True)

            else:
                print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

        else:
            print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

    else:
        print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))

    ### save trajectory PDB
    final_plddt = get_best_plddt(af_model, length)
    af_model.save_pdb(model_pdb_path)
    af_model.aux["log"]["terminate"] = ""

    # let's check whether the trajectory is worth optimising by checking confidence, clashes, and contacts
    # check clashes
    #clash_interface = calculate_clash_score(model_pdb_path, 2.4)
    ca_clashes = calculate_clash_score(model_pdb_path, 2.5, only_ca=True)

    #if clash_interface > 25 or ca_clashes > 0:
    if ca_clashes > 0:
        af_model.aux["log"]["terminate"] = "Clashing"
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        os.remove(model_pdb_path)
        delete_files_with_design_name(design_name, design_paths)
        print("")
    else:
        # check if low quality prediction
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            os.remove(model_pdb_path)
            delete_files_with_design_name(design_name, design_paths)
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 7 contacts then protein is floating above and is not binder
            if binder_contacts_n < 7:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                os.remove(model_pdb_path)
                delete_files_with_design_name(design_name, design_paths)
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    return af_model

# Define radius of gyration loss for colabdesign
def add_rg_loss(self, weight=0.1):
    '''add radius of gyration loss'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:,residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]
        rg = jnp.sqrt(jnp.square(ca - ca.mean(0)).sum(-1).mean() + 1e-8)
        rg_th = 2.38 * ca.shape[0] ** 0.365

        rg = jax.nn.elu(rg - rg_th)
        return {"rg":rg}

    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["rg"] = weight

# Define interface pTM loss for colabdesign
def add_i_ptm_loss(self, weight=0.1):
    def loss_iptm(inputs, outputs):
        p = 1 - get_ptm(inputs, outputs, interface=True)
        i_ptm = mask_loss(p)
        return {"i_ptm": i_ptm}
    
    self._callbacks["model"]["loss"].append(loss_iptm)
    self.opt["weights"]["i_ptm"] = weight

# add helicity loss
def add_helix_loss(self, weight=0):
    def binder_helicity(inputs, outputs):  
      if "offset" in inputs:
        offset = inputs["offset"]
      else:
        idx = inputs["residue_index"].flatten()
        offset = idx[:,None] - idx[None,:]

      # define distogram
      dgram = outputs["distogram"]["logits"]
      dgram_bins = get_dgram_bins(outputs)
      mask_2d = np.outer(np.append(np.zeros(self._target_len), np.ones(self._binder_len)), np.append(np.zeros(self._target_len), np.ones(self._binder_len)))

      x = _get_con_loss(dgram, dgram_bins, cutoff=6.0, binary=True)
      if offset is None:
        if mask_2d is None:
          helix_loss = jnp.diagonal(x,3).mean()
        else:
          helix_loss = jnp.diagonal(x * mask_2d,3).sum() + (jnp.diagonal(mask_2d,3).sum() + 1e-8)
      else:
        mask = offset == 3
        if mask_2d is not None:
          mask = jnp.where(mask_2d,mask,0)
        helix_loss = jnp.where(mask,x,0.0).sum() / (mask.sum() + 1e-8)

      return {"helix":helix_loss}
    self._callbacks["model"]["loss"].append(binder_helicity)
    self.opt["weights"]["helix"] = weight

# add N- and C-terminus distance loss
def add_termini_distance_loss(self, weight=0.1, threshold_distance=7.0):
    '''Add loss penalizing the distance between N and C termini'''
    def loss_fn(inputs, outputs):
        xyz = outputs["structure_module"]
        ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]
        ca = ca[-self._binder_len:]  # Considering only the last _binder_len residues

        # Extract N-terminus (first CA atom) and C-terminus (last CA atom)
        n_terminus = ca[0]
        c_terminus = ca[-1]

        # Compute the distance between N and C termini
        termini_distance = jnp.linalg.norm(n_terminus - c_terminus)

        # Compute the deviation from the threshold distance using ELU activation
        deviation = jax.nn.elu(termini_distance - threshold_distance)

        # Ensure the loss is never lower than 0
        termini_distance_loss = jax.nn.relu(deviation)
        return {"NC": termini_distance_loss}

    # Append the loss function to the model callbacks
    self._callbacks["model"]["loss"].append(loss_fn)
    self.opt["weights"]["NC"] = weight

def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

######################################
### parse input paths
parser = argparse.ArgumentParser(description='Script to run BindCraft binder design.')

parser.add_argument('--settings', '-s', type=str, required=True,
                    help='Path to the basic settings.json file. Required.')
parser.add_argument('--filters', '-f', type=str, default='./settings_filters/default_filters.json',
                    help='Path to the filters.json file used to filter design. If not provided, default will be used.')
parser.add_argument('--advanced', '-a', type=str, default='./settings_advanced/4stage_multimer.json',
                    help='Path to the advanced.json file with additional design settings. If not provided, default will be used.')

args = parser.parse_args()

### load settings from JSON
with open(args.settings, 'r') as file:
    target_settings = json.load(file)

with open(args.advanced, 'r') as file:
    advanced_settings = json.load(file)

with open(args.filters, 'r') as file:
    filters = json.load(file)

settings_file = os.path.basename(args.settings).split('.')[0]
filters_file = os.path.basename(args.filters).split('.')[0]
advanced_file = os.path.basename(args.advanced).split('.')[0]

### load AF2 model settings
# if advanced_settings["use_multimer_design"]:
#     design_models = [0,1,2,3,4]
#     prediction_models = [0,1]
#     multimer_validation = False
# else:
#     design_models = [0,1]
#     prediction_models = [0,1,2,3,4]
#     multimer_validation = True

design_models = [0,1,2,3,4]
prediction_models = [0,1]
multimer_validation = False
advanced_settings["af_params_dir"] = '/work/lpdi/users/mpacesa/Pipelines/BindCraft/'
advanced_settings["dssp_path"] = '/work/lpdi/users/mpacesa/Pipelines/BindCraft/functions/dssp'
advanced_settings["dalphaball_path"] = '/work/lpdi/users/mpacesa/Pipelines/BindCraft/functions/DAlphaBall.gcc'

### generate directories, design path names can be found within the function
design_path_names = ["Trajectory", "Trajectory/Frames"]
design_paths = {}

# make directories and set design_paths[FOLDER_NAME] variable
for name in design_path_names:
    path = os.path.join(target_settings["design_path"], name)
    os.makedirs(path, exist_ok=True)
    design_paths[name] = path

if advanced_settings["omit_AAs"] in [None, False, '']:
    advanced_settings["omit_AAs"] = None
elif isinstance(advanced_settings["omit_AAs"], str):
    advanced_settings["omit_AAs"] = advanced_settings["omit_AAs"].strip()

####################################
####################################
####################################
### start design loop
while True:
    ### Initialise design
    # generate random seed to vary designs
    seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])

    # sample binder design length randomly from defined distribution
    samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
    length = np.random.choice(samples)

    # load desired helicity value to sample different secondary structure contents
    if advanced_settings["random_helicity"] is True:
        # will sample a random bias towards helicity
        helicity_value = round(np.random.uniform(-3, 1),2)
    elif advanced_settings["weights_helicity"] != 0:
        # using a preset helicity bias
        helicity_value = advanced_settings["weights_helicity"]
    else:
        # no bias towards helicity
        helicity_value = 0

    # generate design name and check if same trajectory was already run
    design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
    trajectory_dirs = ["Trajectory"]
    trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)

    if not trajectory_exists:
        print("Starting trajectory: "+design_name)

        ### Begin binder hallucination
        trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths)
        print("")