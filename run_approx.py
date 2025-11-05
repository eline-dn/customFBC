# is our approximation useful?

from functions import *


#from rosetta_functions import *
import os
from metrics_utils import *
import pandas as pd
import glob
import time
#from cofolding_utils import *

# Check if JAX-capable GPU is available, otherwise exit
check_jax_gpu()

######################################
### parse input paths
"""
run this script with the following args: settings_path, filters_path, advanced_path
"""

# perform checks of input setting files, reduced number of iterations
#settings_path, filters_path, advanced_path = "/work/lpdi/users/eline/FastBC/FastBC/settings_target/n9test.json", "/work/lpdi/users/eline/FastBC/FastBC/settings_filters/default_filters.json", "/work/lpdi/users/eline/FastBC/FastBC/settings_advanced/default_hardtarget_n9test.json"

settings_path=sys.argv[1]
filters_path=sys.argv[2]
advanced_path=sys.argv[3]

### load settings from JSON
target_settings, advanced_settings, filters = load_json_settings(settings_path, filters_path, advanced_path)

settings_file = os.path.basename(settings_path).split('.')[0]
filters_file = os.path.basename(filters_path).split('.')[0]
advanced_file = os.path.basename(advanced_path).split('.')[0]

### load AF2 model settings
design_models, prediction_models, multimer_validation = load_af2_models(advanced_settings["use_multimer_design"])

bindcraft_folder="/work/lpdi/users/eline/FastBC/customFBC"
advanced_settings = perform_advanced_settings_check(target_settings, advanced_settings, bindcraft_folder)

### generate directories, design path names can be found within the function
design_paths = generate_directories(target_settings["design_path"])

### generate dataframes
trajectory_labels, design_labels, final_labels = generate_dataframe_labels()
trajectory_csv = os.path.join(target_settings["design_path"], 'trajectory_stats.csv')
mpnn_csv = os.path.join(target_settings["design_path"], 'mpnn_design_stats.csv')
final_csv = os.path.join(target_settings["design_path"], 'final_design_stats.csv')

create_dataframe(trajectory_csv, trajectory_labels)
create_dataframe(mpnn_csv, design_labels)
create_dataframe(final_csv, final_labels)

trajectory_metrics={}

# Initialize an empty DataFrame to store results
all_design_stats_df = pd.DataFrame()
all_design_stats_df.to_csv(os.path.join(target_settings["design_path"], 'all_design_stats.csv'), index=False)

# run a few binders (100), random length

i=0
while i < 100:
  i+=1
  seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
  samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
  length = np.random.choice(samples)
  helicity_value = load_helicity(advanced_settings)
  design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
  trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
  trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)
      # halucinate a binder and extract metrics
  trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                                target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                                  design_models, advanced_settings, design_paths)
  trajectory_metrics[design_name] = copy_dict(trajectory._tmp["best"]["aux"]["log"]) # permet de récupérer iptm plugged et empty
  trajectory_pdb = os.path.join(design_paths["Trajectory"], design_name + ".pdb")
  trajectory.save_pdb(trajectory_pdb)
  trajectory_metrics[design_name] = {k: round(v, 2) if isinstance(v, float) else v for k, v in trajectory_metrics[design_name].items()}

      # Extract binder sequence from .pdb
  trajectory_sequence = trajectory.get_seq(get_best=True)[0]
  binder_sequence = trajectory_sequence # Assuming trajectory_sequence is the binder sequence
  length = len(binder_sequence)

  prediction_model = compile_prediction_models(hardtarget_mode=False, data_dir=advanced_settings["af_params_dir"])

      # Define output folder for templates and predictions
  output_folder = design_paths["Trajectory"] # Or another suitable folder

  pdb_path = trajectory_pdb # Use the generated trajectory PDB
  empty_target_path_specific = extract_template_path(pdb_path, empty=True, hardtarget_mode=False, design_name=design_name, output_folder=output_folder)
  plugged_target_path_specific = extract_template_path(pdb_path, empty=False, hardtarget_mode=False, design_name=design_name, output_folder=output_folder)


      # Run re-predictions with the specific templates
  specific_empty_prediction_stats_df=run_prediction_with_template(model=prediction_model,
                                                          template=empty_target_path_specific, # Use the specific template path
                                                          binder_len=length,
                                                          hardtarget_mode=False,
                                                          binder_sequence=binder_sequence,
                                                          output_folder=output_folder,
                                                          empty=True,
                                                          BC_complex_pdb=pdb_path,
                                                          binder_name=design_name) # Use design_name here
  specific_plugged_prediction_stats_df=run_prediction_with_template(model=prediction_model,
                                                            template=plugged_target_path_specific, # Use the specific template path
                                                            binder_len=length,
                                                            hardtarget_mode=False,
                                                            binder_sequence=binder_sequence,
                                                            output_folder=output_folder,
                                                            empty=False,
                                                            BC_complex_pdb=pdb_path,
                                                            binder_name=design_name) # Use design_name here

      # Combine trajectory metrics and prediction stats into a single dictionary for the current design
  current_design_data = trajectory_metrics[design_name].copy()
  current_design_data.update(specific_empty_prediction_stats_df.iloc[0].to_dict())
  current_design_data.update(specific_plugged_prediction_stats_df.iloc[0].to_dict())

      # Convert the dictionary to a DataFrame row
  current_design_df = pd.DataFrame([current_design_data], index=[design_name])
  print(current_design_df)
  print( list(current_design_df.columns))
  first=True if (i==0) else False
  current_design_df.to_csv(os.path.join(target_settings["design_path"], 'all_design_stats.csv'), mode='a', header=first, index=True)

      # Append the row to the main DataFrame
  #all_design_stats_df = pd.concat([all_design_stats_df, current_design_df])



#end of loop
#print(all_design_stats_df)
#all_design_stats_df.to_csv(os.path.join(target_settings["design_path"], 'all_design_stats.csv'))


#plot 2 metrics

# After the loop, you can save the final DataFrame to a CSV file
# all_design_stats_df.to_csv(os.path.join(target_settings["design_path"], 'all_design_stats.csv'))
