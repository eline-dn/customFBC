####################################
################ BioPython functions
####################################
### Import dependencies
import os
import math
import numpy as np
from collections import defaultdict
from scipy.spatial import cKDTree
from Bio import BiopythonWarning
from Bio.PDB import PDBParser, DSSP, Selection, Polypeptide, PDBIO, Select, Chain, Superimposer
from Bio.PDB.SASA import ShrakeRupley
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio.PDB.Selection import unfold_entities
from Bio.PDB.Polypeptide import is_aa

from Bio.PDB import PDBParser, Selection
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.Polypeptide import is_aa
import numpy as np

# Minimal 3-letter -> 1-letter, including common alt codes
three_to_one_map = {
    "ALA":"A","CYS":"C","ASP":"D","GLU":"E","PHE":"F","GLY":"G","HIS":"H","ILE":"I",
    "LYS":"K","LEU":"L","MET":"M","ASN":"N","PRO":"P","GLN":"Q","ARG":"R","SER":"S",
    "THR":"T","VAL":"V","TRP":"W","TYR":"Y",
    # common variants
    "MSE":"M",  # Selenomethionine
}

def _copy_structure_with_only_chain(structure, chain_id):
    """Return a new Structure containing only model 0 and a deep copy of chain `chain_id`."""
    # Build a tiny structure hierarchy: Structure -> Model(0) -> Chain(chain_id) -> Residues/Atoms
    from Bio.PDB import StructureBuilder
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
        if not is_aa(res, standard=False):
            continue
        hetflag, resseq, icode = res.id
        sb.init_residue(res.resname, hetflag, resseq, icode)
        for atom in res:
            sb.init_atom(atom.name, atom.coord, atom.bfactor, atom.occupancy,
                         atom.altloc, atom.fullname, element=atom.element)
    return sb.get_structure()

def _residue_sasa_map(structure):
    """Compute per-residue SASA (Å^2) using ShrakeRupley; returns {(chain_id, (resseq, icode)): sasa}."""
    sr = ShrakeRupley()  # default probe radius 1.4 Å, 100 points per atom
    sr.compute(structure, level="R")
    out = {}
    model0 = structure[0]
    for chain in model0:
        for res in chain:
            if not is_aa(res, standard=False):
                continue
            hetflag, resseq, icode = res.id
            out[(chain.id, (resseq, icode))] = getattr(res, "sasa", 0.0)
    return out

def _residue_list(chain):
    """Return list of residues that are amino acids (keeps insertion codes)."""
    return [res for res in chain if is_aa(res, standard=False)]

MAX_ASA_TIEN = {
    'A': 121.0, 'R': 265.0, 'N': 187.0, 'D': 187.0, 'C': 148.0,
    'Q': 214.0, 'E': 214.0, 'G': 97.0,  'H': 216.0, 'I': 195.0,
    'L': 191.0, 'K': 230.0, 'M': 203.0, 'F': 228.0, 'P': 154.0,
    'S': 143.0, 'T': 163.0, 'W': 264.0, 'Y': 255.0, 'V': 165.0
}

# Rosetta-like hydrophobic set: apolar + aromatics (Cys excluded; add 'C' if you want)
HYDROPHOBIC_SET = set(['A','C', 'V','I','L','M','P','F','W','Y'])

def score_interface(pdb_file, binder_chain="B"):
    """
    Biopython analogue of the Rosetta score_interface using rASA for surface definition.

    Returns:
      interface_scores (dict), interface_AA (dict[AA->count at interface]), interface_residues_pdb_ids_str (comma-separated).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_file)

    # --- Collect interface residues using your hotspot_residues helper ---
    interface_residues_set = hotspot_residues(pdb_file, binder_chain=binder_chain, atom_distance_cutoff=4.0)

    # Build AA counts and id list
    interface_AA = {aa: 0 for aa in 'ACDEFGHIKLMNPQRSTVWY'}
    interface_residues_pdb_ids = []
    for pdb_res_num, aa_type in interface_residues_set.items():
        if aa_type in interface_AA:
            interface_AA[aa_type] += 1
        interface_residues_pdb_ids.append(f"{binder_chain}{pdb_res_num}")
    interface_residues_pdb_ids_str = ",".join(interface_residues_pdb_ids)
    interface_nres = len(interface_residues_pdb_ids)

    # --- Interface hydrophobicity on the binder (percent hydrophobic at interface) ---
    hydrophobic_aa = HYDROPHOBIC_SET
    hydrophobic_count = sum(interface_AA.get(aa, 0) for aa in hydrophobic_aa)
    interface_hydrophobicity = (hydrophobic_count / interface_nres * 100.0) if interface_nres else 0.0

    # --- Compute SASA: binder-alone (unbound) and binder-in-complex ---
    binder_only = _copy_structure_with_only_chain(structure, binder_chain)
    sasa_unbound_map = _residue_sasa_map(binder_only)   # {(chain_id, (resseq, icode)): ASA}
    sasa_complex_map = _residue_sasa_map(structure)

    binder_sasa_unbound = sum(v for (cid, _), v in sasa_unbound_map.items() if cid == binder_chain)
    binder_sasa_in_complex = sum(v for (cid, _), v in sasa_complex_map.items() if cid == binder_chain)

    interface_dSASA = max(0.0, binder_sasa_unbound - binder_sasa_in_complex)
    interface_binder_fraction = (interface_dSASA / binder_sasa_unbound * 100.0) if binder_sasa_unbound > 0 else 0.0

    # --- Surface residues of binder (unbound) and surface hydrophobicity via rASA ---
    SURF_RASA_CUTOFF = 0.20  # 25% rASA ~ LayerSelector-like surface

    binder_chain_unbound = binder_only[0][binder_chain]
    binder_residues = [res for res in binder_chain_unbound if is_aa(res, standard=False)]

    def reskey(res):
        _, resseq, icode = res.id
        return (resseq, icode)

    surface_total_count = 0
    surface_hydrophobic_count = 0

    for res in binder_residues:
        key = reskey(res)
        asa = sasa_unbound_map.get((binder_chain, key), 0.0)

        resname3 = res.get_resname().strip()
        aa = three_to_one_map.get(resname3)
        if aa is None:
            continue

        max_asa = MAX_ASA_TIEN.get(aa)
        if not max_asa or max_asa <= 0:
            continue

        rASA = asa / max_asa
        if rASA >= SURF_RASA_CUTOFF:
            surface_total_count += 1
            if aa in HYDROPHOBIC_SET:
                surface_hydrophobic_count += 1

    surface_hydrophobicity = (surface_hydrophobic_count / surface_total_count) if surface_total_count else 0.0

    # --- Pack results like your Rosetta dict (only requested fields filled) ---
    interface_scores = {
        'binder_score': None,
        'surface_hydrophobicity': surface_hydrophobicity,     # fraction 0–1
        'interface_sc': None,
        'interface_packstat': None,
        'interface_dG': None,
        'interface_dSASA': interface_dSASA,                   # Å^2
        'interface_dG_SASA_ratio': None,
        'interface_fraction': interface_binder_fraction,      # % of unbound SASA buried
        'interface_hydrophobicity': interface_hydrophobicity, # % hydrophobic at interface
        'interface_nres': interface_nres,
        'interface_interface_hbonds': None,
        'interface_hbond_percentage': None,
        'interface_delta_unsat_hbonds': None,
        'interface_delta_unsat_hbonds_percentage': None
    }

    interface_scores = {k: round(v, 2) if isinstance(v, float) else v
                        for k, v in interface_scores.items()}

    return interface_scores, interface_AA, interface_residues_pdb_ids_str


def unaligned_rmsd(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
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

def align_pdbs(reference_pdb, align_pdb, reference_chain_id, align_chain_id):
    """
    Aligns `align_pdb` onto `reference_pdb` using Cα atoms of the specified chains.
    Overwrites `align_pdb` with the aligned coordinates and then calls `clean_pdb(align_pdb)`.

    Parameters
    ----------
    reference_pdb : str
        Path to the reference PDB file.
    align_pdb : str
        Path to the PDB file that will be transformed/aligned (overwritten).
    reference_chain_id : str
        Chain ID in the reference structure; if comma-separated, only the first is used.
    align_chain_id : str
        Chain ID in the structure to be aligned; if comma-separated, only the first is used.
    """
    # If the chain IDs contain commas, split them and only take the first value
    reference_chain_id = reference_chain_id.split(',')[0].strip()
    align_chain_id = align_chain_id.split(',')[0].strip()

    parser = PDBParser(QUIET=True)
    ref_struct = parser.get_structure("ref", reference_pdb)
    mov_struct = parser.get_structure("mov", align_pdb)

    # Use first model (index 0) for both
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

    # Build resseq -> CA atom maps for standard residues
    def chain_ca_map(chain):
        ca_map = {}
        for res in chain:
            # Skip hetero/water; only standard amino acids
            if not is_aa(res, standard=True):
                continue
            if "CA" in res:
                resseq = res.get_id()[1]  # (hetero flag, resseq, icode) -> take numerical resseq
                # If there are insertion codes, you could include res.get_id()[2] too,
                # but for most cases resseq is enough to match.
                ca_map[(resseq, res.get_id()[2])] = res["CA"]
        return ca_map

    ref_ca = chain_ca_map(ref_chain)
    mov_ca = chain_ca_map(mov_chain)

    # Intersect by (resseq, icode)
    common_keys = sorted(set(ref_ca.keys()).intersection(mov_ca.keys()),
                         key=lambda k: (k[0], (k[1] or " ")))

    if len(common_keys) < 3:
        raise ValueError(
            f"Not enough matching residues between chains {reference_chain_id} (ref) and "
            f"{align_chain_id} (mov) to compute a reliable superposition (found {len(common_keys)})."
        )

    fixed_atoms = [ref_ca[k] for k in common_keys]
    moving_atoms = [mov_ca[k] for k in common_keys]

    # Superimpose
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    # Apply transform to ALL atoms in the moving structure
    rotation, translation = sup.rotran
    for atom in mov_struct.get_atoms():
        atom.transform(rotation, translation)

    # Overwrite aligned pdb
    io = PDBIO()
    io.set_structure(mov_struct)
    io.save(align_pdb)

# temporary function, calculate RMSD of input PDB and trajectory target
def target_pdb_rmsd(trajectory_pdb, starting_pdb, chain_ids_string):
    # Parse the PDB files
    parser = PDBParser(QUIET=True)
    structure_trajectory = parser.get_structure('trajectory', trajectory_pdb)
    structure_starting = parser.get_structure('starting', starting_pdb)
    
    # Extract chain A from trajectory_pdb
    chain_trajectory = structure_trajectory[0]['A']
    
    # Extract the specified chains from starting_pdb
    chain_ids = chain_ids_string.split(',')
    residues_starting = []
    for chain_id in chain_ids:
        chain_id = chain_id.strip()
        chain = structure_starting[0][chain_id]
        for residue in chain:
            if is_aa(residue, standard=True):
                residues_starting.append(residue)
    
    # Extract residues from chain A in trajectory_pdb
    residues_trajectory = [residue for residue in chain_trajectory if is_aa(residue, standard=True)]
    
    # Ensure that both structures have the same number of residues
    min_length = min(len(residues_starting), len(residues_trajectory))
    residues_starting = residues_starting[:min_length]
    residues_trajectory = residues_trajectory[:min_length]
    
    # Collect CA atoms from the two sets of residues
    atoms_starting = [residue['CA'] for residue in residues_starting if 'CA' in residue]
    atoms_trajectory = [residue['CA'] for residue in residues_trajectory if 'CA' in residue]
    
    # Calculate RMSD using structural alignment
    sup = Superimposer()
    sup.set_atoms(atoms_starting, atoms_trajectory)
    rmsd = sup.rms
    
    return round(rmsd, 2)

# analyze sequence composition of design
def validate_design_sequence(sequence, num_clashes, advanced_settings):
    note_array = []

    # Check if protein contains clashes after relaxation
    if num_clashes > 0:
        note_array.append('Relaxed structure contains clashes.')

    # Check if the sequence contains disallowed amino acids
    if advanced_settings["omit_AAs"]:
        restricted_AAs = advanced_settings["omit_AAs"].split(',')
        for restricted_AA in restricted_AAs:
            if restricted_AA in sequence:
                note_array.append('Contains: '+restricted_AA+'!')

    # Analyze the protein
    analysis = ProteinAnalysis(sequence)

    # Calculate the reduced extinction coefficient per 1% solution
    extinction_coefficient_reduced = analysis.molar_extinction_coefficient()[0]
    molecular_weight = round(analysis.molecular_weight() / 1000, 2)
    extinction_coefficient_reduced_1 = round(extinction_coefficient_reduced / molecular_weight * 0.01, 2)

    # Check if the absorption is high enough
    if extinction_coefficient_reduced_1 <= 2:
        note_array.append(f'Absorption value is {extinction_coefficient_reduced_1}, consider adding tryptophane to design.')

    # Join the notes into a single string
    notes = ' '.join(note_array)

    return notes

# detect C alpha clashes for deformed trajectories
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