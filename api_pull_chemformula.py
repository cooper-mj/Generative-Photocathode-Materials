import json
import requests
import pandas as pd
import numpy as np
from mendeleev import element
from periodictable import formulas

materials_codes = open("Raw_dataset/full_mpid_list_cell_size_sort.txt")

url_base = "https://www.materialsproject.org/rest/v1/materials/"
url_tail = "/vasp/pretty_formula?API_KEY="
api_key = "TsOqaHtJ0LRPqf5jL2Hn"

output_path = "data/unit_cell_data.csv"
formula_output_path = "data/chemical_formulas.txt"

# features based on averages over all atoms in the chemical formula
features = ["MPID", "atomic_number", "atomic_weight", "atomic_radius", "density", "en_pauling", "lattice_constant"]

all_formulas = []
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
		feats[0] = elem.atomic_number
		feats[1] = elem.atomic_weight
		feats[2] = elem.atomic_radius
		feats[3] = elem.density
		feats[4] = elem.en_pauling
		feats[5] = elem.lattice_constant
		# print(feats)
		averages += count * feats
		# print(totals)
	averages /= total_atoms
	print(averages)
	return averages

i = 0
for code in materials_codes:
	material = code.rstrip() # Add a string for it
	url = url_base + material + url_tail + api_key
	
	print("Iteration " + str(i + 1) + "/10000")

	response = requests.get(url)
	data = response.json()
	if len(data["response"]) == 0: # No data for this material available
		continue

	material_features = np.zeros(len(features))
	material_features[0] = float(code[3:])

	# Getting atoms from chemical formula
	chem_formula = data["response"][0]["pretty_formula"]
	all_formulas.append(chem_formula)	

	# Getting average features over all atoms in formula
	averages = extract_features(chem_formula, material_features, len(features) - 1)
	material_features[1:] = averages

	features_all_elements.append(material_features)

	i += 1

	# if i > 5:
	# 	break

print(all_formulas)

# Write stuff to outuput files
df = pd.DataFrame(data=features_all_elements, columns=features)
print(df)
df.to_csv(path_or_buf=output_path, index=False)

with open(formula_output_path, 'w') as f:
    for formula in all_formulas:
        f.write("%s\n" % formula)