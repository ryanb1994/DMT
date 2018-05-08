import csv
import time

start = time.time()

with open('training.csv', 'r') as csvfile:
	f = csv.reader(csvfile)
	skip = next(f)

	# You can see how it works for the first row:
	
	# i = 0

	# for row in f:
	# 	if i == 0: 
	# 		print row
	# 		print row[15]
	# 		print '\n'
	# 		i = 1

end = time.time()

print end - start