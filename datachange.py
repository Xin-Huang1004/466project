import numpy
import random

lines = numpy.loadtxt('Skin_NonSkin.txt')
result =[]
# get 50000 skin data
for i in range(0,50000):
	if(lines[i][3] == 1):
		lines[i][3] = 1
	elif(lines[i][3] == 2):
		lines[i][3] = 0
	else:
		pass
	result.append(lines[i])
# get 50000 Non-skin data
for i in range(60000,110000):
	if(lines[i][3] == 1):
		lines[i][3] = 1
	elif(lines[i][3] == 2):
		lines[i][3] = 0
	else:
		pass
	result.append(lines[i])

index_list = []
for i in range(len(result)):
    index_list.append(i)

numpy.random.shuffle(index_list)
output = []
for item in range(len(result)):
     output.append(result[index_list[item]])
              
numpy.savetxt("result_1.txt",output,fmt='%d',newline='\n')

