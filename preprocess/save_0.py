import os
import numpy as np

pdb_folder = './preprocess/data_negative/'
pdb_files = [f for f in os.listdir(pdb_folder) if f.endswith('.pdb')]

pdb_names = [os.path.splitext(f)[0] for f in pdb_files]

print(f"Found {len(pdb_names)} PDB files without extension.")

pdb_names_array = np.array(pdb_names)


np.save('./features/ABPDB_label_0.npy', pdb_names_array)

loaded_names = np.load('./features/ABPDB_label_0.npy')
print(f"Loaded {len(loaded_names)} PDB names from npy file.")
print(loaded_names)