


# loading libraries
from rosetta_functions import *
import os
from metrics_utils import *
import pandas as pd
import glob



def run_whole_reprediction(binder_name, binder_sequence, target_path,empty:bool,output_folder,BC_complex_pdb, params ):
  models=[0,1]

  af_model = colabdesign.mk_af_model(protocol='binder',
                    use_multimer=True,
                    use_templates=False,
                    data_dir=params,
                   )
  fasta=(binder_name, binder_sequence)
  binder_sequence = fasta[1].strip()
  des_name = fasta[0][1:].strip()

  af_model.prep_inputs(pdb_filename=target_path,
               chain='A',
               binder_len=len(binder_sequence)
              )
  prediction_stats = {}
  for model_num in [0,1]:
      af_model.predict(seq=binder_sequence,
                    models=[model_num],
                    num_recycles=3,
                    num_models=1)
      

      barrel="empty" if empty else "plugged"

      os.makedirs(f"{output_folder}/whole_reprediction_models", exist_ok=True)
      predicted_folder=f"{output_folder}/whole_reprediction_models"
      predicted_complex_pdb = os.path.join(predicted_folder, f"{binder_name}_model_{model_num+1}_whole_rep_{barrel}.pdb")

      af_model.save_pdb(predicted_complex_pdb)
      prediction_metrics = copy_dict(af_model.aux["log"]) # contains plddt, ptm, i_ptm, pae, i_pae

            # extract the statistics for the model
      stats = {
                f"wrepred_{barrel}_pLDDT": round(prediction_metrics['plddt'], 2),
                f"wrepred_{barrel}_pTM": round(prediction_metrics['ptm'], 2),
                f"wrepred_{barrel}_i_pTM": round(prediction_metrics['i_ptm'], 2),
                f"wrepred_{barrel}_pAE": round(prediction_metrics['pae'], 2),
                f"wrepred_{barrel}_i_pAE": round(prediction_metrics['i_pae'], 2)
            }


        #  RMSD calculate to determine if binder is in the designed binding site
      rmsd_site = unaligned_rmsd(BC_complex_pdb, predicted_complex_pdb, "B", "B")
      stats[f"wrepred_{barrel}_Binder_RMSD_to_binding_site"] = rmsd_site # this should be used to filter the models that are binding in the predicted binding site

      prediction_stats[model_num+1] = stats # 2 dictionnaries index 1 and 2 to eventually add to the metrics df
  transformed_df = transform_prediction_stats_to_df(prediction_stats)
  return transformed_df
