from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import csv
import time
from matplotlib import cm

start = time.time()

stars_reviews = []
sorted_stars_reviews = []
number_of = []


def string_to_float(l):
	if len(l) > 1:
		if l[0] == '0' and l[2] == '0':
			return 0.0
		if l[0] == '0' and l[2] == '5':
			return 0.5
		if l[0] == '1' and l[2] == '0':
			return 1.0
		if l[0] == '1' and l[2] == '5':
			return 1.5
		if l[0] == '2' and l[2] == '0':
			return 2.0
		if l[0] == '2' and l[2] == '5':
			return 2.5
		if l[0] == '3' and l[2] == '0':
			return 3.0
		if l[0] == '3' and l[2] == '5':
			return 3.5
		if l[0] == '4' and l[2] == '0':
			return 4.0
		if l[0] == '4' and l[2] == '5':
			return 4.5
		if l[0] == '5' and l[2] == '0':
			return 5.0
	return 0.0
	
def out_of_loop():
	global sorted_stars_reviews
	global stars_reviews
	global i

	for k in range(0, len(sorted_stars_reviews)):
		if stars_reviews[i][0] == sorted_stars_reviews[k][0] and stars_reviews[i][1] == sorted_stars_reviews[k][1]:
			number_of[k] += 1
			return
		if k == len(sorted_stars_reviews) - 1:
			if stars_reviews[i][0] == sorted_stars_reviews[k][0] and stars_reviews[i][1] == sorted_stars_reviews[k][1]:
				number_of[k] += 1
				return
			else:
				sorted_stars_reviews.append(stars_reviews[i])
				number_of.append(1)
				return

with open('training.csv', 'r') as csvfile:
	f = csv.reader(csvfile)
	skip = next(f)

	for row in f:
		if row[53] == '1': 
			stars_reviews.append([int(row[8]), string_to_float(row[9])])

	for i in range(0, len(stars_reviews) - 1):
		if i == 0:
			sorted_stars_reviews.append(stars_reviews[i])
			number_of.append(1)
		if i > 1:
			out_of_loop()

xpos = []
ypos = []

for row in sorted_stars_reviews:
	xpos.append(row[0])
	ypos.append(row[1])

zpos = np.ones(len(sorted_stars_reviews))

dx = np.ones(len(sorted_stars_reviews))
dy = np.ones(len(sorted_stars_reviews))
dz = number_of

print 'Plotting now the 3d histogram...'

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

colors = plt.cm.jet(dz/np.amax(dz))

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color=colors)

plt.show()

end = time.time()
print end - start
