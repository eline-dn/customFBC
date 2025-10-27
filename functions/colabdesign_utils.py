####################################
############## ColabDesign functions
####################################
### Import dependencies
import os, re, shutil, math, pickle
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import softmax
from colabdesign import mk_afdesign_model, clear_mem
from colabdesign.mpnn import mk_mpnn_model
from colabdesign.af.alphafold.common import residue_constants
from colabdesign.af.loss import get_ptm, mask_loss, get_dgram_bins, _get_con_loss, get_plddt_loss, get_exp_res_loss, get_pae_loss, get_con_loss, get_rmsd_loss, get_dgram_loss, get_fape_loss
from colabdesign.shared.utils import copy_dict
from colabdesign.shared.prep import prep_pos
from .biopython_utils import hotspot_residues, calculate_clash_score, calc_ss_percentage, calculate_percentages, align_pdbs, unaligned_rmsd, score_interface

# hallucinate a binder
def binder_hallucination(design_name, starting_pdb, chain, target_hotspot_residues, length, seed, helicity_value, design_models, advanced_settings, design_paths):
    model_pdb_path = os.path.join(design_paths["Trajectory"], design_name+".pdb")

    # clear GPU memory for new trajectory
    clear_mem()

    # initialise binder hallucination model
    af_model = mk_afdesign_model(protocol="binder", debug=False, data_dir=advanced_settings["af_params_dir"], 
                                use_multimer=advanced_settings["use_multimer_design"], num_recycles=advanced_settings["num_recycles_design"],
                                best_metric='loss')

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

    # Normalize num_pos
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
    #advanced_settings["use_empty_i_ptm_loss"]=True
    #advanced_settings["weights_empty_iptm"]=0.5
    if advanced_settings["use_empty_i_ptm_loss"]:
        # interface pTM loss
        add_empty_i_ptm_loss(af_model, advanced_settings["weights_empty_iptm"])

    if advanced_settings["use_termini_distance_loss"]:
        # termini distance loss
        add_termini_distance_loss(af_model, advanced_settings["weights_termini_distance_loss"])

    if advanced_settings["use_termini_angle_loss"]:
        # termini angle loss
        add_termini_angle_loss(af_model, advanced_settings["weights_termini_angle_loss"])

    if advanced_settings["cyclize_peptide"]:
        # make macrocycle peptide
        add_cyclic_offset(af_model)

    # add the helicity loss
    add_helix_loss(af_model, helicity_value)

    # calculate the number of mutations to do based on the length of the protein
    greedy_tries = math.ceil(length * (advanced_settings["greedy_percentage"] / 100))

    ### start design algorithm based on selection
    if advanced_settings["design_algorithm"] == 'default':
        print("Stage 1: Test Logits")
        af_model.design_logits(iters=advanced_settings["test_iterations"], e_soft=advanced_settings["test_softmaxing"], models=design_models, num_models=1, sample_models=advanced_settings["sample_models"], save_best=True)

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
            if advanced_settings["soft_iterations"] > 0:
                print("Stage 1: Additional Logits Optimisation")
                af_model.clear_best()
                af_model.design_logits(iters=advanced_settings["soft_iterations"], e_soft=1, models=design_models, num_models=1, sample_models=advanced_settings["sample_models"],
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
                        af_model.design_semigreedy(iters=advanced_settings["greedy_iterations"], tries=greedy_tries, dropout=False, save_best=True,
                                                    seq_logits=af_model.aux["seq"]["logits"], e_tries=None, models=design_models, num_models=1,
                                                    sample_models=advanced_settings["sample_models"])

                else:
                    #update_failures(failure_csv, 'Trajectory_one-hot_pLDDT')
                    print("One-hot trajectory pLDDT too low to continue: "+str(onehot_plddt))

            else:
                #update_failures(failure_csv, 'Trajectory_softmax_pLDDT')
                print("Softmax trajectory pLDDT too low to continue: "+str(softmax_plddt))

        else:
            #update_failures(failure_csv, 'Trajectory_logits_pLDDT')
            print("Initial trajectory pLDDT too low to continue: "+str(initial_plddt))
    # end BindCraft default design model
    
    elif advanced_settings["design_algorithm"] == 'mcmc':
        # design by using random mutations that decrease loss
        half_life = round(advanced_settings["greedy_iterations"] / 5, 0)
        t_mcmc = 0.01
        af_model._design_mcmc(advanced_settings["greedy_iterations"], half_life=half_life, T_init=t_mcmc, mutation_rate=greedy_tries, num_models=1, models=design_models,
                                sample_models=advanced_settings["sample_models"], save_best=True)

    else:
        print("ERROR: No valid design model selected")
        exit()
        return

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
        #update_failures(failure_csv, 'Trajectory_Clashes')
        print("Severe clashes detected, skipping analysis and MPNN optimisation")
        print("")
    else:
        # check if low quality prediction
        if final_plddt < 0.7:
            af_model.aux["log"]["terminate"] = "LowConfidence"
            #update_failures(failure_csv, 'Trajectory_final_pLDDT')
            print("Trajectory starting confidence low, skipping analysis and MPNN optimisation")
            print("")
        else:
            # does it have enough contacts to consider?
            binder_contacts = hotspot_residues(model_pdb_path)
            binder_contacts_n = len(binder_contacts.items())

            # if less than 3 contacts then protein is floating above and is not binder
            if binder_contacts_n < 3:
                af_model.aux["log"]["terminate"] = "LowConfidence"
                #update_failures(failure_csv, 'Trajectory_Contacts')
                print("Too few contacts at the interface, skipping analysis and MPNN optimisation")
                print("")
            else:
                # phew, trajectory is okay! We can continue
                af_model.aux["log"]["terminate"] = ""
                print("Trajectory successful, final pLDDT: "+str(final_plddt))

    # move low quality prediction:
    if af_model.aux["log"]["terminate"] != "":
        shutil.move(model_pdb_path, design_paths[f"Trajectory/{af_model.aux['log']['terminate']}"])

    ### get the sampled sequence for plotting
    af_model.get_seqs()
    if advanced_settings["save_design_trajectory_plots"]:
        plot_trajectory(af_model, design_name, design_paths)

    ### save the hallucination trajectory animation
    if advanced_settings["save_design_animations"]:
        plots = af_model.animate(dpi=150)
        with open(os.path.join(design_paths["Trajectory/Animation"], design_name+".html"), 'w') as f:
            f.write(plots)
        plt.close('all')

    if advanced_settings["save_trajectory_pickle"]:
        with open(os.path.join(design_paths["Trajectory/Pickle"], design_name+".pickle"), 'wb') as handle:
            pickle.dump(af_model.aux['all'], handle, protocol=pickle.HIGHEST_PROTOCOL)

    return af_model

# run prediction for binder with masked template target
def predict_binder_complex(prediction_model, binder_sequence, mpnn_design_name, target_pdb, chain, length, trajectory_pdb, prediction_models, advanced_settings, filters, design_paths, seed=None):
    prediction_stats = {}

    # clean sequence
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())

    # reset filtering conditionals
    pass_af2_filters = True
    filter_failures = {}

    if advanced_settings["cyclize_peptide"]:
        # make macrocycle peptide
        add_cyclic_offset(prediction_model)

    # start prediction per AF2 model, 2 are used by default due to masked templates
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(complex_pdb):
            # predict model
            prediction_model.predict(seq=binder_sequence, models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(complex_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'i_pTM': round(prediction_metrics['i_ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2), 
                'i_pAE': round(prediction_metrics['i_pae'], 2)
            }
            prediction_stats[model_num+1] = stats

            # List of filter conditions and corresponding keys
            filter_conditions = [
                (f"{model_num+1}_pLDDT", 'plddt', '>='),
                (f"{model_num+1}_pTM", 'ptm', '>='),
                (f"{model_num+1}_i_pTM", 'i_ptm', '>='),
                (f"{model_num+1}_pAE", 'pae', '<='),
                (f"{model_num+1}_i_pAE", 'i_pae', '<='),
            ]

            # perform initial AF2 values filtering to determine whether to skip relaxation and interface scoring
            for filter_name, metric_key, comparison in filter_conditions:
                threshold = filters.get(filter_name, {}).get("threshold")
                if threshold is not None:
                    if comparison == '>=' and prediction_metrics[metric_key] < threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1
                    elif comparison == '<=' and prediction_metrics[metric_key] > threshold:
                        pass_af2_filters = False
                        filter_failures[filter_name] = filter_failures.get(filter_name, 0) + 1

            if not pass_af2_filters:
                break

    # AF2 filters passed, contuing with relaxation
    for model_num in prediction_models:
        complex_pdb = os.path.join(design_paths["MPNN"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if pass_af2_filters:
            mpnn_relaxed = os.path.join(design_paths["MPNN/Relaxed"], f"{mpnn_design_name}_model{model_num+1}.pdb")
            shutil.copy(complex_pdb, mpnn_relaxed)
            #pr_relax(complex_pdb, mpnn_relaxed)
        else:
            if os.path.exists(complex_pdb):
                os.remove(complex_pdb)

    return prediction_stats, pass_af2_filters

# run prediction for binder alone
def predict_binder_alone(prediction_model, binder_sequence, mpnn_design_name, length, trajectory_pdb, binder_chain, prediction_models, advanced_settings, design_paths, seed=None):
    binder_stats = {}

    # prepare sequence for prediction
    binder_sequence = re.sub("[^A-Z]", "", binder_sequence.upper())
    prediction_model.set_seq(binder_sequence)

    if advanced_settings["cyclize_peptide"]:
        # make macrocycle peptide
        add_cyclic_offset(prediction_model)

    # predict each model separately
    for model_num in prediction_models:
        # check to make sure prediction does not exist already
        binder_alone_pdb = os.path.join(design_paths["MPNN/Binder"], f"{mpnn_design_name}_model{model_num+1}.pdb")
        if not os.path.exists(binder_alone_pdb):
            # predict model
            prediction_model.predict(models=[model_num], num_recycles=advanced_settings["num_recycles_validation"], verbose=False)
            prediction_model.save_pdb(binder_alone_pdb)
            prediction_metrics = copy_dict(prediction_model.aux["log"]) # contains plddt, ptm, pae

            # align binder model to trajectory binder
            align_pdbs(trajectory_pdb, binder_alone_pdb, binder_chain, "A")

            # extract the statistics for the model
            stats = {
                'pLDDT': round(prediction_metrics['plddt'], 2), 
                'pTM': round(prediction_metrics['ptm'], 2), 
                'pAE': round(prediction_metrics['pae'], 2)
            }
            binder_stats[model_num+1] = stats

    return binder_stats

# run MPNN to generate sequences for binders
def mpnn_gen_sequence(trajectory_pdb, binder_chain, trajectory_interface_residues, advanced_settings):
    # clear GPU memory
    clear_mem()

    # initialise MPNN model
    mpnn_model = mk_mpnn_model(backbone_noise=advanced_settings["backbone_noise"], model_name=advanced_settings["model_path"], weights=advanced_settings["mpnn_weights"])

    # check whether keep the interface generated by the trajectory or whether to redesign with MPNN
    design_chains = 'A,' + binder_chain

    if advanced_settings["mpnn_fix_interface"]:
        fixed_positions = 'A,' + trajectory_interface_residues
        fixed_positions = fixed_positions.rstrip(",")
        print("Fixing interface residues: "+trajectory_interface_residues)
    else:
        fixed_positions = 'A'

    # prepare inputs for MPNN
    mpnn_model.prep_inputs(pdb_filename=trajectory_pdb, chain=design_chains, fix_pos=fixed_positions, rm_aa=advanced_settings["omit_AAs"])

    # sample MPNN sequences in parallel
    mpnn_sequences = mpnn_model.sample(temperature=advanced_settings["sampling_temp"], num=1, batch=advanced_settings["num_seqs"])

    return mpnn_sequences

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

# Get pLDDT of best model
def get_best_plddt(af_model, length):
    return round(np.mean(af_model._tmp["best"]["aux"]["plddt"][-length:]),2)

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

# Define  custom empty iptm loss for colabdesign

def add_empty_i_ptm_loss(self, weight=0.1):
    target_len = self._target_len
    target_trim=24
    binder_len=self._binder_len
    def custom_empty_iptm_loss(inputs, outputs):
    """
    Custom loss to compute ipTM for a binder + trimmed target complex.

    Parameters
    ----------
    inputs : dict
        Standard model input dict (includes batch, params, opt, etc.)
    outputs : dict
        Model output dict containing AlphaFold predictions.
    target_trim : int, optional
        Number of N-terminal residues to remove from the target structure (default: 24)

    Returns
    -------
    loss_value : float
        ipTM value computed from the binder + trimmed target complex.
    """
      # === 1. Extract sequences ===
      # Get target sequence from inputs["batch"]["aatype"] and convert to string
      from colabdesign.af.alphafold.common import residue_constants
      aatype_target = np.asarray(inputs["batch"]["aatype"][:target_len])
      seq_target = "".join([residue_constants.restypes_with_x[i] for i in aatype_target])
      print("extracted target sequence")

    # Get binder sequence from model parameters
      aatype_full = inputs["aatype"]
      total_len = aatype_full.shape[0]
      binder_aatype = aatype_full[target_len:total_len]
      binder_seq = "".join([residue_constants.restypes_with_x[i] for i in binder_aatype])
      #If your binder length is known (e.g., binder_len), you could slice: binder_aatype = aatype_full[target_len:target_len+binder_len].
      if np.all(binder_aatype == 0):
        print("All zeros in binder, binder not ready yet!")


      binder_len=len(binder_seq)
      print(f"extracted binder sequence: {binder_seq}, length={binder_len}")
  ######## debugged until here, need to find the outputs first :(
      # ==== 2. extract trimmed pdb structure ===
      structure = outputs_to_biopython_structure(outputs, seq_target, target_len, chain_id="A")
      trimmed_pdb = trim_structure(structure, n_trim=target_trim)
      print("extracted empty pdb, running reprediction")
      # === reprediction and ipTM computation ===
      # run a new prediction with binder_seq + trimmed_target_pdb as template,
      # and compute the ipTM score.
      clear_mem()
      model=mk_afdesign_model(protocol="binder",
                          num_recycles=3,
                          data_dir=advanced_settings["af_params_dir"],
                          use_multimer=True,
                          use_initial_guess=False,
                          use_template=True,
                          use_initial_atom_pos=False )

      model.prep_inputs(pdb_filename=trimmed_pdb,
                          chain="A",
                          #binder_chain="B",# do not specifiy if the template only contains the target
                          binder_len=binder_len,
                          rm_target_seq=False, #b
                          use_binder_template=False, #a
                          rm_template_ic=False #c
                          )
      prediction_stats = {}
      for model_num in [0,1]:
        model.predict(seq=binder_sequence,
                      models=[model_num],
                      num_recycles=3)
        ipTM[model_num+1] = model.aux["log"]["ipTM"]


      empty_iptm = (ipTM[1] + ipTM[2]) / 2

    # === 3. Return the custom loss value (negate if minimizing) ===
      print(f"successful empty iptm extraction: {empty_iptm}" )
      return {"empty_i_ptm":1-iptm_value}

    self._callbacks["model"]["loss"].append(loss_empty_iptm)
    self.opt["weights"]["empty_i_ptm"] = weight

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

# enable design of cyclic peptides, by Sergey Ovchinnikov
def add_cyclic_offset(self, offset_type=2):
    '''add cyclic offset to connect N and C term'''
    def cyclic_offset(L):
        i = np.arange(L)
        ij = np.stack([i,i+L],-1)
        offset = i[:,None] - i[None,:]
        c_offset = np.abs(ij[:,None,:,None] - ij[None,:,None,:]).min((2,3))

        if offset_type == 1:
            c_offset = c_offset

        elif offset_type >= 2:
            a = c_offset < np.abs(offset)
            c_offset[a] = -c_offset[a]

        if offset_type == 3:
            idx = np.abs(c_offset) > 2
            c_offset[idx] = (32 * c_offset[idx] )/  abs(c_offset[idx])

        return c_offset * np.sign(offset)

    idx = self._inputs["residue_index"]
    offset = np.array(idx[:,None] - idx[None,:])
    c_offset = cyclic_offset(self._binder_len)
    offset[self._target_len:,self._target_len:] = c_offset

    self._inputs["offset"] = offset

def add_termini_angle_loss(self, weight=0.1):
   """
   Add loss penalizing the angle between two vectors:
   1. Target center to binder center.
   2. Binder center to NC middlepoint.
   """
   def loss_fn(inputs, outputs):
       xyz = outputs["structure_module"]
       ca = xyz["final_atom_positions"][:, residue_constants.atom_order["CA"]]

       # Assuming target and binder are concatenated in ca:
       target_ca = ca[:-self._binder_len]  # Target CA atoms
       binder_ca = ca[-self._binder_len:]  # Binder CA atoms

       # Calculate centers of geometry
       target_center = jnp.mean(target_ca, axis=0)
       binder_center = jnp.mean(binder_ca, axis=0)

       # Calculate N and C termini of binder
       binder_n_terminus = binder_ca[0]
       binder_c_terminus = binder_ca[-1]

       # Calculate midpoint of binder's N and C termini
       binder_nc_midpoint = (binder_n_terminus + binder_c_terminus) / 2.0

       # Calculate vectors
       vector1 = binder_center - target_center
       vector2 = binder_nc_midpoint - binder_center

       # Normalize vectors to unit length
       vector1_norm = vector1 / jnp.linalg.norm(vector1)
       vector2_norm = vector2 / jnp.linalg.norm(vector2)

       # Calculate cosine of the angle between vectors
       cos_angle = jnp.dot(vector1_norm, vector2_norm)

       # Calculate the angle in radians (0 to pi)
       angle = jnp.arccos(jnp.clip(cos_angle, -1.0, 1.0))

       # Penalize deviations from 0 (perfect alignment)
       angle_loss = angle  # or angle**2 to penalize larger deviations more

       return {"termini_angle": angle_loss}

   # Append the loss function to the model callbacks
   self._callbacks["model"]["loss"].append(loss_fn)
   self.opt["weights"]["angle"] = weight

# plot design trajectory losses
def plot_trajectory(af_model, design_name, design_paths):
    metrics_to_plot = ['loss', 'plddt', 'ptm', 'i_ptm', 'con', 'i_con', 'pae', 'i_pae', 'rg', 'mpnn']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for index, metric in enumerate(metrics_to_plot):
        if metric in af_model.aux["log"]:
            # Create a new figure for each metric
            plt.figure()

            loss = af_model.get_loss(metric)
            # Create an x axis for iterations
            iterations = range(1, len(loss) + 1)

            plt.plot(iterations, loss, label=f'{metric}', color=colors[index % len(colors)])

            # Add labels and a legend
            plt.xlabel('Iterations')
            plt.ylabel(metric)
            plt.title(design_name)
            plt.legend()
            plt.grid(True)

            # Save the plot
            plt.savefig(os.path.join(design_paths["Trajectory/Plots"], design_name+"_"+metric+".png"), dpi=150)
            
            # Close the figure
            plt.close()



# add the functions for the custom loss:

from Bio.PDB import PDBIO

def trim_structure(structure, n_trim=24):
    for model in structure:
        for chain in model:
            # Get all residues in the chain
            residues = list(chain.get_residues())
            # Keep residues from index trim_length onwards
            for i, residue in enumerate(residues):
                if i < trim_length:
                    chain.detach_child(residue.get_id())
    output_pdb_path("empty_template.pdb")
    io = PDBIO(use_model_flag=1)
    io.set_structure(structure)
    io.save(output_pdb_path)
    return(output_pdb_path)

# Example usage:
# Assuming you have a PDB file named 'input.pdb' in the current directory
# trim_pdb('input.pdb', 'trimmed_output.pdb')
# print("Trimmed PDB file saved as 'trimmed_output.pdb'")


from Bio.PDB import StructureBuilder
import numpy as np
from colabdesign.af.alphafold.common import residue_constants

def outputs_to_biopython_structure(outputs, seq_target, target_len, chain_id="A"):
    """
    Convert AlphaFold outputs (target part) to a Bio.PDB Structure.

    Parameters
    ----------
    outputs : dict
        Model outputs containing structure_module keys.
    seq_target : str
        Target sequence (length == target_len).
    target_len : int
        Number of residues belonging to the target.
    chain_id : str, optional
        Chain ID for the Bio.PDB structure (default: 'A').

    Returns
    -------
    structure : Bio.PDB.Structure.Structure
        Biopython structure containing the target coordinates.
    """

    atom_positions = np.asarray(outputs["structure_module"]["final_atom_positions"])
    atom_mask = np.asarray(outputs["structure_module"]["final_atom_mask"])

    # Only keep target part
    atom_positions = atom_positions[:target_len]
    atom_mask = atom_mask[:target_len]

    # Build the structure
    builder = StructureBuilder.StructureBuilder()
    builder.init_structure("AF_Target")
    builder.init_model(1)
    builder.init_chain(chain_id)

    for i, res in enumerate(seq_target):
        resname = residue_constants.restype_3to1.get(res, res)
        builder.init_residue(res, " ", i + 1, " ")

        atom_names = residue_constants.atom_types
        coords = atom_positions[i]
        mask = atom_mask[i]

        for atom_name, coord, m in zip(atom_names, coords, mask):
            if m > 0.5:  # only add valid atoms
                builder.init_atom(atom_name, coord, 1.0, 1.0, " ", atom_name, i + 1, element=atom_name[0])

    structure = builder.get_structure()
    return structure
