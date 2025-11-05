### these are the functions used in running_metrics.py
import os
import pandas as pd
import colabdesign
import sys
import numpy as np
import jax
from colabdesign.shared.utils import copy_dict
from Bio.PDB import PDBParser, PDBIO

from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import DSSP, Selection, Polypeptide, Select, Chain, Superimposer
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities


from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB import PDBParser, PDBIO, Model, Chain, Structure
from Bio.PDB import StructureBuilder
from Bio.PDB.Polypeptide import is_aa # Assuming is_aa is needed and available

import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss, get_plddt_loss, get_exp_res_loss, get_pae_loss, get_con_loss, get_rmsd_loss, get_dgram_loss, get_fape_loss
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.prep import prep_pos
#from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages, align_pdbs, unaligned_rmsd, score_interface


def load_bindcraft_metrics(df, design_name: str, required_cols = [
        "Design", "Sequence","InterfaceResidues","Average_pLDDT", "Average_pTM", "Average_i_pTM",
        "Average_pAE", "Average_i_pAE", "Average_i_pLDDT", "Average_ss_pLDDT",
        "Average_dSASA", "Average_Interface_SASA_%", "Average_Interface_Hydrophobicity",
        "Average_n_InterfaceResidues", "Average_Binder_pLDDT", "Average_Binder_pTM",
        "Average_Binder_pAE", "Average_Binder_RMSD"]) -> dict:



    # Ensure expected columns exist

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in BindCraft CSV")

    # Find the row corresponding to the design name
    row = df.loc[df["Design"] == design_name]
    if row.empty:
        print(f"[Warning] Design '{design_name}' not found in {csv_path}")
        return {}

    # Convert that row (Series) to a dictionary, keeping only the metrics
    row_dict = row.iloc[0][required_cols[:]].to_dict()

    # Add the design name as a key for reference
    #row_dict["Design"] = design_name

    return row_dict

    #see functions.py file for pyRosetta functions

def load_bindcraft_metrics_bis(df, design_name: str, required_cols) -> dict:



    # Ensure expected columns exist

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing expected column '{col}' in BindCraft CSV")

    # Find the row corresponding to the design name
    row = df.loc[df["DesignName"] == design_name]
    if row.empty:
        print(f"[Warning] Design '{design_name}' not found in {csv_path}")
        return {}

    # Convert that row (Series) to a dictionary, keeping only the metrics
    row_dict = row.iloc[0][required_cols[:]].to_dict()

    # Add the design name as a key for reference
    #row_dict["Design"] = design_name

    return row_dict



def compile_prediction_models(hardtarget_mode,data_dir): #-> prediction_model
    """ 
    mk_afdesign_model for case 1 and 2, i.e. binder reprediction from the binder sequence and the target structure
    hardtarget_mode:bool
    data_dir:str, where are stored the af models?
    """ 
    clear_mem()
    complex_prediction_model=mk_afdesign_model(protocol="binder",
                        num_recycles=3,
                        data_dir=data_dir,
                        use_multimer=True,
                        use_initial_guess=hardtarget_mode,
                        use_templates=True,
                        use_initial_atom_pos=False )
    return complex_prediction_model



def extract_template_path(BC_complex_pdb:str, hardtarget_mode:bool, empty: bool,output_folder:str, design_name:str): #extract target with the right set of rotamers for binding
    """ output of this function shoulb be used as an input for the run_prediction_with_template function if we wish to use the target in the right rotamer configuration for each binder.
    can be trimmed or not, or used in hardtarget mode or not
    In the case of repredicting the binder in the non specific target configuration (target used as an input for BindCraft eg.), do not use this function and directly use the BC target
    as an input for run_prediction_with_template()

    complex_pdb:str the path to the complex .pdb file. will be trimmed or not, and the binder will be kept or not
    hardtarget_mode:bool if set to True, template_pdb will also contain the binder
    outputfolder: where do you want to save the generated templates
    """

    os.makedirs(f"{output_folder}/specific_templates", exist_ok=True)
    pdb_folder=f"{output_folder}/specific_templates"

    if hardtarget_mode:
      pdb=BC_complex_pdb # prediction will use target + binder, keep the original pdb

    else: # extract chain "A" only,
       pdb=f"{pdb_folder}/{design_name}_specific_target.pdb"
       extract_chain(BC_complex_pdb, pdb, chain_id="A") # still need to chack this function

      # empty: True or False (then return plugged target)
    if empty:
      trimmed_pdb=f"{pdb_folder}/{design_name}_empty_target.pdb"
      trim_pdb(pdb, trimmed_pdb, trim_length=24)
      return trimmed_pdb
    else:
      return pdb # return the template pdb, without trimming


def run_prediction_with_template(model,
                                     template,
                                     binder_len,
                                     hardtarget_mode,
                                     binder_sequence,
                                     output_folder:str,
                                     empty:bool,
                                     BC_complex_pdb,
					binder_name:str,
				     specific=True):
    """
    hadtarget_mode:bool, same purpose as in BC, sets a and c to True in that case, the template will have to contains also the binder structure
    if set to False, template should only contain the target structure
    output_folder: folder to store the repredicted pdb folder
    BC_complex_pdb:str structure of the BC predicted complex, should be be the same as the one used as an input to extract_template_path
    specific:bool, is the template used specific to the binder (beta barrel in the right binding rotamer configuration)
    """
    if hardtarget_mode:
      model.prep_inputs(pdb_filename=template,
                        chain="A",
                        binder_chain="B",# do not specifiy if the template only contains the target
                        binder_len=binder_len,
                        rm_target_seq=False, #b
                        use_binder_template=hardtarget_mode, #a
                        rm_template_ic=hardtarget_mode #c
                        )
    else:
      model.prep_inputs(pdb_filename=template,
                        chain="A",
                        #binder_chain="B",# do not specifiy if the template only contains the target
                        binder_len=binder_len,
                        rm_target_seq=False, #b
                        use_binder_template=hardtarget_mode, #a
                        rm_template_ic=hardtarget_mode #c
                        )
    prediction_stats = {}
    for model_num in [0]:
      model.predict(seq=binder_sequence,
                    models=[model_num],
                    num_recycles=3)
      barrel="empty" if empty else "plugged"
      spec="specific_" if specific else ""
      os.makedirs(f"{output_folder}/predicted_models", exist_ok=True)
      predicted_folder=f"{output_folder}/predicted_models"
      predicted_complex_pdb = os.path.join(predicted_folder, f"{spec}{binder_name}_model_{model_num+1}_repredicted_{barrel}.pdb")
      model.save_pdb(predicted_complex_pdb)
      prediction_metrics = copy_dict(model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

      # extract the statistics for the model
      stats = {
          f"{spec}{barrel}_pLDDT": round(prediction_metrics['plddt'], 2),
          f"{spec}{barrel}_pTM": round(prediction_metrics['ptm'], 2),
          f"{spec}{barrel}_i_pTM": round(prediction_metrics['i_ptm'], 2),
          f"{spec}{barrel}_pAE": round(prediction_metrics['pae'], 2),
          f"{spec}{barrel}_i_pAE": round(prediction_metrics['i_pae'], 2)
            }


      # unaligned RMSD calculate to determine if binder is in the designed binding site
      rmsd_site = unaligned_rmsd(BC_complex_pdb, predicted_complex_pdb, "B", "B")
      stats[f"{spec}{barrel}_Binder_RMSD_to_binding_site"] = rmsd_site # this should be used to filter the models that are binding in the predicted binding site

      prediction_stats[model_num+1] = stats # 2 dictionnaries index 1 and 2 to eventually add to the metrics df
    transformed_df = transform_prediction_stats_to_df(prediction_stats)
    return transformed_df




def transform_prediction_stats_to_df(prediction_stats: dict) -> pd.DataFrame:
    """
    Transforms the prediction_stats dictionary into a single-row pandas DataFrame.

    Args:
        prediction_stats (dict): A dictionary where keys are model numbers (1 or 2)
                                 and values are dictionaries of statistics.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns in the format
                      'model_num_statistique' and corresponding values.
    """
    transformed_data = {}
    for model_num, stats in prediction_stats.items():
        for stat_name, stat_value in stats.items():
            transformed_data[f"{model_num}_{stat_name}"] = stat_value

    return pd.DataFrame([transformed_data])

# Example usage (assuming prediction_stats is defined as in your code):
# prediction_stats = {
#     1: {'pLDDT': 90.5, 'pTM': 0.8, 'i_pTM': 0.75, 'pAE': 10.2, 'i_pAE': 12.5, 'Binder_RMSD_to_binding_site': 1.5},
#     2: {'pLDDT': 88.2, 'pTM': 0.78, 'i_pTM': 0.72, 'pAE': 11.0, 'i_pAE': 13.1, 'Binder_RMSD_to_binding_site': 2.1}
# }
# transformed_df = transform_prediction_stats_to_df(prediction_stats)
# display(transformed_df)


def _copy_structure_with_only_chain(structure, chain_id):
    """Return a new Structure containing only model 0 and a deep copy of chain `chain_id`."""
    # Build a tiny structure hierarchy: Structure -> Model(0) -> Chain(chain_id) -> Residues/Atoms

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("single")
    sb.init_model(1)
    sb.init_chain(chain_id)
    # Set segment ID, padded to 4 characters
    sb.init_seg(chain_id.ljust(4))    
    model0 = structure[0]
    if chain_id not in [c.id for c in model0.get_chains()]:
        raise ValueError(f"Chain '{chain_id}' not found.")
    chain = model0[chain_id]
    for res in chain:
        # Keep only amino-acid residues
        # Assuming is_aa is defined elsewhere and available
        if not is_aa(res, standard=False):
            continue
        hetflag, resseq, icode = res.id
        sb.init_residue(res.resname, hetflag, resseq, icode)

        for atom in res:
            sb.init_atom(atom.name, atom.coord, atom.bfactor, atom.occupancy,
                         atom.altloc, atom.fullname, element=atom.element)
    return sb.get_structure()


def _copy_structure_with_only_chain0(structure, chain_id): # wrong version
    """Return a new Structure containing only model 0 and a deep copy of chain `chain_id`."""
    # Build a tiny structure hierarchy: Structure -> Model(0) -> Chain(chain_id) -> Residues/Atoms

    sb = StructureBuilder.StructureBuilder()
    sb.init_structure("single")
    sb.init_model(0)
    sb.init_chain(chain_id)
    model0 = structure[0]
    if chain_id not in [c.id for c in model0.get_chains()]:
        raise ValueError(f"Chain '{chain_id}' not found.")
    chain = model0[chain_id]
    for res in chain:
        # Keep only amino-acid residues
        # Assuming is_aa is defined elsewhere and available
        if not is_aa(res, standard=False):
            continue
        hetflag, resseq, icode = res.id
        sb.init_residue(res.resname, hetflag, resseq, icode)
        for atom in res:
            sb.init_atom(atom.name, atom.coord, atom.bfactor, atom.occupancy,
                         atom.altloc, atom.fullname, element=atom.element)
    return sb.get_structure()


def extract_chain(input_pdb_path: str, output_pdb_path: str, chain_id: str):
    """
    Extracts a specific chain from a PDB file using _copy_structure_with_only_chain
    and saves it to a new PDB file with explicit MODEL/ENDMDL records.

    Args:
        input_pdb_path (str): Path to the input PDB file (complex).
        output_pdb_path (str): Path to save the extracted chain PDB file.
        chain_id (str): The identifier of the chain to extract (e.g., "A", "B").
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)
    io = PDBIO(use_model_flag=1)

    # Use the helper function to get a new structure with only the desired chain
    new_structure = _copy_structure_with_only_chain(structure, chain_id)

    # --- Debug Print Statements ---
    print(f"--- Debug: Saving structure for {output_pdb_path} ---")
    print(f"Number of models in structure to save: {len(new_structure)}")
    for i, model in enumerate(new_structure):
        print(f"  Model {i}: Number of chains = {len(model)}")
        for j, chain in enumerate(model):
            print(f"    Chain {chain.id}: Number of residues = {len(chain)}")
            # Optional: print a few residue IDs and segids to confirm content
            print(f"      First few residues (ID, SegID): {[(r.id, r.segid) for r in list(chain.get_residues())[:5]]}")
    print("----------------------------------------------------")
    # --- End Debug Print Statements ---

    # Save the new structure, explicitly writing model records
    io.set_structure(new_structure)
    io.save(output_pdb_path)

# Example usage (assuming you have a complex PDB file named 'complex.pdb'):
# extract_chain('complex.pdb', 'chain_A.pdb', 'A')
# print("Chain A saved to 'chain_A.pdb'")

def extract_chain01(input_pdb_path: str, output_pdb_path: str, chain_id: str): # same model error
    """
    Extracts a specific chain from a PDB file using _copy_structure_with_only_chain
    and saves it to a new PDB file.
	
    Args:
        input_pdb_path (str): Path to the input PDB file (complex).
        output_pdb_path (str): Path to save the extracted chain PDB file.
        chain_id (str): The identifier of the chain to extract (e.g., "A", "B").
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)
    io = PDBIO()

    # Use the helper function to get a new structure with only the desired chain
    new_structure = _copy_structure_with_only_chain(structure, chain_id)

    # Save the new structure
    io.set_structure(new_structure)
    io.save(output_pdb_path)

# Example usage (assuming you have a complex PDB file named 'complex.pdb'):
# extract_chain('complex.pdb', 'chain_A.pdb', 'A')
# print("Chain A saved to 'chain_A.pdb'")



def extract_chain0(input_pdb_path: str, output_pdb_path: str, chain_id: str): # not working
    """
    Extracts a specific chain from a PDB file and saves it to a new PDB file.

    Args:
        input_pdb_path (str): Path to the input PDB file (complex).
        output_pdb_path (str): Path to save the extracted chain PDB file.
        chain_id (str): The identifier of the chain to extract (e.g., "A", "B").
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)
    io = PDBIO()

    class SelectChain:
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_model(self, model):
            # Accept all models
            return True

        def accept_chain(self, chain):
            # Accept only the specified chain
            return chain.get_id() == self.chain_id

        def accept_residue(self, residue):
            # Accept all residues within the selected chain
            return True

        def accept_atom(self, atom):
            # Accept all atoms within the selected residues
            return True
    io.set_structure(structure)
    io.save(output_pdb_path, SelectChain(chain_id))

# Example usage (assuming you have a complex PDB file named 'complex.pdb'):
# extract_chain('complex.pdb', 'chain_A.pdb', 'A')
# print("Chain A saved to 'chain_A.pdb'")


def trim_pdb(input_pdb_path, output_pdb_path, trim_length=26):
    """
    Trims the first N amino acids from a PDB file.

    Args:
        input_pdb_path (str): Path to the input PDB file.
        output_pdb_path (str): Path to save the trimmed PDB file.
        trim_length (int): The number of amino acids to trim from the beginning.
    """
    parser = PDBParser()
    structure = parser.get_structure("protein", input_pdb_path)

    for model in structure:
        for chain in model:
            # Get all residues in the chain
            residues = list(chain.get_residues())
            # Keep residues from index trim_length onwards
            for i, residue in enumerate(residues):
                if i < trim_length:
                    chain.detach_child(residue.get_id())

    io = PDBIO(use_model_flag=1)
    io.set_structure(structure)
    io.save(output_pdb_path)

# Example usage:
# Assuming you have a PDB file named 'input.pdb' in the current directory
# trim_pdb('input.pdb', 'trimmed_output.pdb')
# print("Trimmed PDB file saved as 'trimmed_output.pdb'")




def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    unaligned RMSD of binder compared to original trajectory, in other words how far is binder in the repredicted complex from the original binding site
    Compute Cα RMSD between chains in two PDBs *without* superposition.
    Residues are matched by (resseq, insertion code) intersection.

    Parameters
    ----------
    reference_pdb : str
        Path to the reference PDB file.
    align_pdb : str
        Path to the PDB file to compare against.
    reference_chain_id : str
        Chain ID in the reference structure; if comma-separated, only the first is used.
    align_chain_id : str
        Chain ID in the moving structure; if comma-separated, only the first is used.

    Returns
    -------
    float
        RMSD in Å, rounded to 2 decimals.

    Raises
    ------
    ValueError
        If chains are missing or there are fewer than 3 matched residues with Cα atoms.
    """
    # Use first value if comma-separated
    reference_chain_id = reference_chain_id.split(',')[0].strip()
    align_chain_id = align_chain_id.split(',')[0].strip()

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", reference_pdb)
    mov_struct = parser.get_structure("mov", align_pdb)

    ref_model = next(ref_struct.get_models())
    mov_model = next(mov_struct.get_models())

    # Fetch chains
    try:
        ref_chain = ref_model[reference_chain_id]
    except KeyError:
        raise ValueError(f"Reference chain '{reference_chain_id}' not found in {reference_pdb}.")
    try:
        mov_chain = mov_model[align_chain_id]
    except KeyError:
        raise ValueError(f"Align chain '{align_chain_id}' not found in {align_pdb}.")

    # Build maps of Cα atoms keyed by (resseq, icode)
    def ca_map(chain):
        out = {}
        for res in chain:
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                hetflag, resseq, icode = res.get_id()
                out[(resseq, icode)] = res["CA"]
        return out

    ref_ca = ca_map(ref_chain)
    mov_ca = ca_map(mov_chain)

    # Common residue keys, ordered by residue number then insertion code
    common = sorted(set(ref_ca.keys()).intersection(mov_ca.keys()),
                    key=lambda k: (k[0], (k[1] or " ")))

    if len(common) < 3:
        raise ValueError(
            f"Not enough matched residues with Cα to compute RMSD without alignment "
            f"(found {len(common)})."
        )

    ref_coords = np.array([ref_ca[k].get_coord() for k in common], dtype=float)
    mov_coords = np.array([mov_ca[k].get_coord() for k in common], dtype=float)

    # Unaligned RMSD = sqrt(mean(||ref - mov||^2))
    diffs = ref_coords - mov_coords
    rmsd = float(np.sqrt((diffs * diffs).sum(axis=1).mean()))
    return round(rmsd, 2)
