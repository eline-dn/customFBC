from functions import *

# Check if JAX-capable GPU is available, otherwise exit
check_jax_gpu()

######################################
### parse input paths


# perform checks of input setting files
settings_path, filters_path, advanced_path = "/work/lpdi/users/eline/FastBC/FastBC/settings_target/n9test.json", "/work/lpdi/users/eline/FastBC/FastBC/settings_filters/default_filters.json", "/work/lpdi/users/eline/FastBC/FastBC/settings_advanced/default_hardtarget.json" 

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
create_dataframe(final_csv, final_labels)
seed = int(np.random.randint(0, high=999999, size=1, dtype=int)[0])
samples = np.arange(min(target_settings["lengths"]), max(target_settings["lengths"]) + 1)
length = np.random.choice(samples)
helicity_value = load_helicity(advanced_settings)
design_name = target_settings["binder_name"] + "_l" + str(length) + "_s"+ str(seed)
trajectory_dirs = ["Trajectory", "Trajectory/Relaxed", "Trajectory/LowConfidence", "Trajectory/Clashing"]
trajectory_exists = any(os.path.exists(os.path.join(design_paths[trajectory_dir], design_name + ".pdb")) for trajectory_dir in trajectory_dirs)
trajectory = binder_hallucination(design_name, target_settings["starting_pdb"], target_settings["chains"],
                                            target_settings["target_hotspot_residues"], length, seed, helicity_value,
                                            design_models, advanced_settings, design_paths)
print("test ok")
