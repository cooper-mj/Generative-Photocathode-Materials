import json
import requests
import math
import pandas as pd
import numpy as np
from mendeleev import element
from periodictable import formulas

# materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")
chemical_formulas = pd.read_csv('all_chemical_formulas.csv')
num_materials = len(chemical_formulas)

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/pretty_formula?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

output_path = "material_average_data2.csv"

# features based on averages over all atoms in the chemical formula
features = ["MPID", "Atomic Volume", "Atomic weight", "Boiling point", 
			"Covalent Radius",  "Density", "Electron Affinity", 
			"Electronegativity", "Group", "Heat of formation", 
			"Heat of fusion", "Heat of vaporization","Lattice constant", 
			"Melting Point", "Period", "Polarizability", "Specific Heat"]

features_all_elements = []

# Compute average features for one material
def extract_features(atoms, materials_features, num_features):
	atoms = (formulas.formula(chem_formula)).atoms
	averages = np.zeros(num_features)
	total_atoms = 0

	for atom in atoms:
		count = atoms[atom]
		total_atoms += count

		elem = element(atom.symbol)
		feats = np.zeros(num_features)
		feats[0] = elem.atomic_volume if elem.atomic_volume else 0
		feats[1] = elem.atomic_weight if elem.atomic_weight else 0
		feats[2] = elem.boiling_point if elem.boiling_point else 0
		feats[3] = elem.covalent_radius_bragg if elem.covalent_radius_bragg else 0
		feats[4] = elem.density if elem.density else 0
		feats[5] = elem.electron_affinity if elem.electron_affinity else 0
		feats[6] = elem.en_pauling if elem.en_pauling else 0
		feats[7] = elem.group_id if elem.group_id else 0
		feats[8] = elem.heat_of_formation if elem.heat_of_formation else 0
		feats[9] = elem.fusion_heat if elem.fusion_heat else 0
		feats[10] = elem.evaporation_heat if elem.evaporation_heat else 0
		feats[11] = elem.lattice_constant if elem.lattice_constant else 0
		feats[12] = elem.melting_point if elem.melting_point else 0
		feats[13] = elem.period if elem.period else 0
		feats[14] = elem.dipole_polarizability if elem.dipole_polarizability else 0
		feats[15] = elem.specific_heat if elem.specific_heat else 0
		
		# print(feats)
		averages += count * feats

	averages /= total_atoms
	# print(averages)
	return averages

## Pulling from chemical formula txt file
for i in range(num_materials):
	if i % 100 == 0:
		print('Iteration %d' % i)
	material_features = np.zeros(len(features))

	code = chemical_formulas.iloc[i, 0]
	chem_formula = chemical_formulas.iloc[i, 1]
	
	material_features[0] = float(code[3:])
	averages = extract_features(chem_formula, material_features, len(features) - 1)
	material_features[1:] = averages

	features_all_elements.append(material_features)

	# if i > 5:
	# 	break

# # Pulling directly from API
# i = 0
# for code in materials_codes:
# 	material = code.rstrip()
# 	url = url_base + material + url_tail + api_key
# 	print(material)
	
# 	# if i % 10 == 0:
# 	# 	print("Iteration " + str(i + 1) + "/10000")

# 	response = requests.get(url)
# 	data = response.json()
# 	if "error" in data or len(data["response"]) == 0: # no data available for this material
# 		continue

# 	material_features = np.zeros(len(features))
# 	material_features[0] = float(code[3:])

# 	# Getting atoms from chemical formula
# 	chem_formula = data["response"][0]["pretty_formula"]
# 	all_formulas.append(chem_formula)	

# 	# Getting average features over all atoms in formula
# 	averages = extract_features(chem_formula, material_features, len(features) - 1)
# 	material_features[1:] = averages

# 	features_all_elements.append(material_features)

# 	# if i != 0 and i % 100 == 0:
# 	# 	df = pd.DataFrame(data=features_all_elements, columns=features)
# 	# 	# print(df)
# 	# 	df.to_csv(path_or_buf=output_path, index=False)

# 	i += 1


# Write stuff to output files
df = pd.DataFrame(data=features_all_elements, columns=features)
print(df)
df.to_csv(path_or_buf=output_path, index=False)