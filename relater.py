import pandas as pd
import quat2euler


#classes for translation 3
#classes for rotation 5


confactor = 3.14/180.0
outfile = open("total0.csv", "a")
csvfile = "train2.csv"
read = pd.read_csv(csvfile)
print len(read.index)
for index in range(2, len(read.index)):
	res = read.ix[index][:]
	ref = read.ix[index-1][:]
	#df.ix[index][0] = res[0]
	#df.ix[index][1] = res[1]
	#df.ix[index][2] = res[0]
	if ref[0][:17] != res[0][:17]:
		continue
	trancount = 0
	rotcount = 0
	for i in range(1,4):
		val = round(res[i]-ref[i], 1)
		#print val
		if val > 0.0:
			rule = 0
		elif val < 0.0:
			rule = 2
		elif val == 0.0:
			rule = 1
		trancount += rule*pow(3, 3-i)

	for i in range(4,7):
		val = round(res[i]-ref[i], 1)
		#if i ==5:
			#print val
		if val < -0.1:
			rule = 0
		elif val < 0.0:
			rule = 1
		elif val == 0.0:
			rule = 2
		elif val < 0.2:
			rule = 3
		elif val >= 0.2:
			rule = 4
		rotcount += rule*pow(5, 6-i)
	#df.ix[index][2] = trancount
	#print counter
	#df.ix[index][3] = rotcount
	outfile.write(str(ref[0]) + ", " + str(res[0]) + ", " + str(trancount) + ", " + str(rotcount) + "\n")
	#print outrot[0], outrot[1], outrot[2]
	#read.ix[index][5] = outrot[0]
	#read.ix[index][6] = outrot[1]
	#read.ix[index][7] = outrot[2]

	#write translation class


	#write rotation class

#read.drop('qz', axis=0)
