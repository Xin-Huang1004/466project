import numpy

lines = numpy.loadtxt('Skin_NonSkin.txt')
result =[]
# get 50000 skin data
for i in range(0,50000):
	if(lines[i][3] == 1):
		lines[i][3] = 0
	elif(lines[i][3] == 2):
		lines[i][3] = 1
	else:
		pass
	result.append(lines[i])
# get 50000 Non-skin data
for i in range(60000,110000):
	if(lines[i][3] == 1):
		lines[i][3] = 0
	elif(lines[i][3] == 2):
		lines[i][3] = 1
	else:
		pass
	result.append(lines[i])

numpy.savetxt("result_1.txt",result,fmt='%d',newline='\n')

