import pandas as pd
import quat2euler


csvfile = "final.csv"
read = pd.read_csv(csvfile)
print read.index
#df = pd.DataFrame(index = len(read.index))
for index in range(1, 10):
	res = read.ix[index][:]
	#sdf.ix[index][0:5] = res[0:5]
	inquat = [res[5], res[6], res[7], res[8]] 
	#inquat = res[5:8]
	print inquat
	#read.ix[index][:] = read.ix[index][:7]
	outrot = quat2euler.quat2euler(inquat)
	print outrot[0], outrot[2]
	read.ix[index][5] = outrot[0]
	read.ix[index][6] = outrot[1]
	read.ix[index][7] = outrot[2]


read.to_csv("total.csv", sep=',')
read.drop('qz', axis=0)