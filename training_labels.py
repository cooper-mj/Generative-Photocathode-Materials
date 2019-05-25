import pandas as pd
import numpy as np
import math

NUM_MATERIALS = 10000
NUM_MEASUREMENTS = 99 	# excluding first measurement
NO_EMITTANCE = float('inf')

path = "Raw_dataset/"
mpid_file = "full_mpid_list_cell_size_sort.txt"
emittance_file = "Screening_data/intrinsic_emittance_"
output_path = "data/emittance_labels.csv"

info = ["MPID", "min emittance"]

mpids = open(path + mpid_file)
materials_codes = mpids.readlines()
info_all_materials = []

for i in range(NUM_MATERIALS):
# for i in range(5): # USE THIS IF YOU JUST WANT TO TEST IT OUT
	print ("Iteration " + str(i + 1) + "/10000")
	material_info = np.zeros(len(info))

	material = materials_codes[i].rstrip()
	print(material)
	material_info[0] = float(material[3:])

	try:
		emittance = np.loadtxt(path + emittance_file + str(i) + ".txt")[1:]
	except:
		print("Unable to read file " + path + emittance_file + str(i) + ".txt")
		material_info[1] = NO_EMITTANCE
		info_all_materials.append(material_info)
		continue

	# Load txt file and remove first measurement, which is always 0.225
	np.around(emittance, decimals=1)
	# Place measurements into buckets of nearest 0.1

	# Maybe there's a more pythonic way to keep only elements that happen twice
	# in a row, but I sure couldn't find it
	min_emit = -1
	for j in range(NUM_MEASUREMENTS - 1):
		if emittance[j] != 0 and math.isclose(emittance[j], emittance[j+1], rel_tol=0.01):
			if (min_emit == -1 or emittance[j] < min_emit):
				min_emit = emittance[j]

	material_info[1] = min_emit if min_emit > 0 else NO_EMITTANCE
	info_all_materials.append(material_info)

df = pd.DataFrame(data=info_all_materials, columns=info)
print(df)
df.to_csv(path_or_buf=output_path, index=False)
