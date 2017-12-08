import numpy
import random

lines = numpy.loadtxt('Skin_NonSkin.txt')
train_test =[]
validation = []
# randomly shuffle index list
index_list = []
for i in range(len(lines)):
    index_list.append(i)
numpy.random.shuffle(index_list)

# we want to randomly pick 100000 data from Skin_NonSkin.txt
output = []
output1 = []

for item in range(100000):
     output.append(index_list[item])

for item in range(100000,200000):
     output1.append(index_list[item])

# pick data and set target to 0 or 1
for i in (output):
	if(lines[i][3] == 1):
		lines[i][3] = 1
	elif(lines[i][3] == 2):
		lines[i][3] = 0
	else:
		pass
	train_test.append(lines[i])

for i in (output1):
	if(lines[i][3] == 1):
		lines[i][3] = 1
	elif(lines[i][3] == 2):
		lines[i][3] = 0
	else:
		pass
	validation.append(lines[i])

# save output data     
numpy.savetxt("train_test.txt",train_test,fmt='%d',newline='\n')
numpy.savetxt("validation.txt",validation,fmt='%d',newline='\n')


# this is for normalize data for QQ plot
normal = numpy.loadtxt('train_test.txt')
for ii in range(len(normal)):
	for item in range(len(normal[ii])-1):
		normal[ii][item] = normal[ii][item] / 255
numpy.savetxt("normalize.txt",normal,fmt='%f',newline='\n')

