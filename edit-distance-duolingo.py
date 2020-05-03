ref_file = "test.en_pt.2020-01-13.gold.txt"
pred_file = "sys_detok"

prompt_idx = -1
list_of_prompt_refs = []

with open(ref_file, 'r') as infile:
	for line in infile:
		if not line.strip():
			continue #skip empty lines

		t = line.split("|")[0]
		if ("prompt" in line):
			first = True
			prompt_idx += 1
			list_of_prompt_refs.append([])
			continue
		else:
			list_of_prompt_refs[prompt_idx].append(t)

# Test using this code:
# print(list_of_prompt_refs[0])
# print(len(list_of_prompt_refs))
# print(len(list_of_prompt_refs[0]))
# print(len(list_of_prompt_refs[1]))


with open(pred_file) as f:
    lines = f.read().splitlines()
    # print(len(lines))
    # for i in range(10):
    # 	print(lines[i])

from Levenshtein import distance as ldistance
import numpy as np

count = 0
perfect_matches = 0
for hyp, ref_list in zip(lines, list_of_prompt_refs):
	distances = [ldistance(hyp, ref) for ref in ref_list]
	min_dist = min([ldistance(hyp, ref) for ref in ref_list])
	min_idx =  np.argmin(distances)
	# How many predictions were closer to some reference other than the top:
	if min_idx != 0:
		count += 1

	# How many times did the exact match exist in the predictions:
	if min_dist == 0:
		perfect_matches += 1
		
print("# How many predictions were closer to some reference other than the top:")
print("{:.2f}".format(100*count/len(lines)), "%")
print()
print("# How many times did top prediction exact matched one of the references:")
print("{:.2f}".format(100*perfect_matches/len(lines)), "%")