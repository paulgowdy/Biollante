import numpy as np
import pickle
'''
print("Loading means and stds")

with open('data_saves/means_1.p', 'rb') as f:

	means = pickle.load(f)

with open('data_saves/stds_1.p', 'rb') as f:

	stds = pickle.load(f)
'''


def flatten(d):    
	res = []  # Result list
	if isinstance(d, dict):
		for key, val in sorted(d.items()):

			#print(key)
			res.extend(flatten(val))

	elif isinstance(d, list):
		res = d        
	else:
		res = [d]

	return res

def xz_relative_pos(obs_list):

	pelvis_x = obs_list[78]
	pelvis_z = obs_list[80]

	x_ids = [66, 69, 72, 75, 81, 84, 87, 90, 93, 96, 330]
	z_ids = [a + 2 for a in x_ids]

	relative_obs_list = list(obs_list)

	for i in range(len(x_ids)):

		relative_obs_list[x_ids[i]] -= pelvis_x
		relative_obs_list[z_ids[i]] -= pelvis_z

	return relative_obs_list

def process_obs_dict(d):

	# Flatten
	f_obs = flatten(d)

	# Relative X, Z
	rel_obs = xz_relative_pos(f_obs)

	# Append velvec, only use in difficult == 0
	rel_obs.extend([3.0, 0.0, 0.0])

	# Normalize
	rel_obs = np.array(rel_obs)
	#rel_obs -= means
	#rel_obs /= stds

	return rel_obs
