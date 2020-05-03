with open("test.en_hu.2020-01-13.gold.txt", 'r') as infile, open("ref.txt", 'w'), open("top1.txt", 'w') as t1, open("top2.txt", 'w') as t2, open("top3.txt", 'w') as t3, open("top4.txt", 'w') as t4:
	for line in infile:
		t = line.split("|")[0]

		if ("prompt" in line):
			first = True
			continue
		elif first:
			t1.write(f"{t}\n")
			first = False
			second = True
			continue
		elif second:
			t2.write(f"{t}\n")
			second = False
			third = True
			continue
		elif third:
			t3.write(f"{t}\n")
			third = False
			fourth = True
			continue
		elif fourth:
			t4.write(f"{t}\n")
			fourth = False
			continue
		else:
			pass


