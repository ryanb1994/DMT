# -*- coding: utf-8 -*-

# The number of bookings when there is no competition and the price is cheaper at the booking site where we are and there is no hotel available (yes-counter)
# For no-counter there is there are cheaper hotels at other sites and availability
# For equal-counter everything is 0 or NULL. 

import csv
import time

start = time.time()

number_of_bookings = 0
yes_counter = 0
equal_counter = 0
no_counter = 0
i = 0

with open('training.csv', 'rb') as f:
	reader = csv.reader(f)
	for row in f:
		if row[53] == '1':
			number_of_bookings += 1
			if row[26] == '1' or row[27] == '1' or row[29] == '1' or row[30] == '1' or row[32] == '1' or row[33] == '1' or row[35] == '1' or row[36] == '1' or row[38] == '1' or row[39] == '1' or row[41] == '1' or row[42] == '1' or row[44] == '1' or row[45] == '1' or row[47] == '1' or row[48] == '1':
				if not row[26] == '-1' or row[27] == '-1' or row[29] == '-1' or row[30] == '-1' or row[32] == '-1' or row[33] == '-1' or row[35] == '-1' or row[36] == '-1' or row[38] == '-1' or row[39] == '-1' or row[41] == '-1' or row[42] == '-1' or row[44] == '-1' or row[45] == '-1' or row[47] == '-1' or row[48] == '-1':
					yes_counter += 1
			else:				 
				if row[26] == '-1' or row[27] == '-1' or row[29] == '-1' or row[30] == '-1' or row[32] == '-1' or row[33] == '-1' or row[35] == '-1' or row[36] == '-1' or row[38] == '-1' or row[39] == '-1' or row[41] == '-1' or row[42] == '-1' or row[44] == '-1' or row[45] == '-1' or row[47] == '-1' or row[48] == '-1':
					no_counter += 1
				else:
					equal_counter += 1
		i += 1

print '--------------------'
print number_of_bookings
print yes_counter
print equal_counter
print no_counter
print '--------------------'

end = time.time()

print end - start