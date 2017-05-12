# -*- coding: UTF-8 -*-
import pandas as pd 
import numpy as np

def get_I():
    I = []
    iTemp = []
    lines = open('./data/paper.txt')
    preline='s'
    for line in lines:
        line = line.strip('\n')
        if (line == '') and (preline != '') :
            I.append(iTemp)
            iTemp = []
	    preline=line
        else:
            iTemp.append(line)
	    preline = line

    data_paper = pd.DataFrame(I)
    data_paper.to_csv('./data/data_paper.csv',encoding='utf-8')
    return I



# --------------- basis res data -----------------
def get_data1():
    index = '#index'
    paperName = '#*'
    authorNname = '#@'
    time = '#t'
    c = '#c'
    indexN = '#%'
    Idata = []
    Idatatmp = []
    s = ''
    I = get_I()
    for nu in range(len(I)):
	#---------- merge paper name ----------------
	flag = 0
	for i  in  range(len(I[nu])):
		if I[nu][i].startswith(paperName):
			Idatatmp.append(I[nu][i].lstrip('#*'))
			flag = 1
	if flag == 0:
		Idatatmp.append(' ')

	# ----------- merge author name --------------
	flag = 0
	for i in range(len(I[nu])):
		if I[nu][i].startswith(authorNname):
			Idatatmp.append(I[nu][i].lstrip('#@'))
			flag = 1
        if flag == 0:
                Idatatmp.append(' ')
	# ------------ meger time --------------------
        flag = 0
        for i in range(len(I[nu])):
                if  I[nu][i].startswith(time):
                        Idatatmp.append(I[nu][i].lstrip('#t'))
			flag = 1
        if flag == 0:
                Idatatmp.append(' ')

	# ------------ meger #c ---------------------
	flag = 0
        for i in range(len(I[nu])):
                if  I[nu][i].startswith(c):
                        Idatatmp.append(I[nu][i].lstrip('#c'))
                        flag = 1
        if flag == 0:
                Idatatmp.append(' ')	
	#----------- merge indexN -------------------
	flag = 0
	for i in range(len(I[nu])):
		if  I[nu][i].startswith(index):
			Idatatmp.append(I[nu][i].lstrip('#index'))
			flag = 1
	if flag == 0:
		Idatatmp.append(' ')
		
	#---------- merge citation  index for every paper ----------------
	flag = 0
	for i in range(len(I[nu])):
		if  I[nu][i].startswith(indexN):
			s = s + I[nu][i].lstrip('#%') + ','
			flag = 1
	if flag == 0:
		Idatatmp.append(' ')
		s = ''
	else :
		Idatatmp.append(s)
		s = ''
	# ------------ Idata -----------------------
	Idata.append(Idatatmp)
	Idatatmp = []
    data_I = pd.DataFrame(Idata)
    data_I.to_csv('./data/data_I',index=False,encoding='utf-8')
    return Idata    
# ---------------------
data = []
datatmp = []
Idata = get_data1()
for nu in range(len(Idata)):
        if nu % 10000 == 0:
                print ('finish -- %d / %d') %(nu,len(I))
        for name in Idata[nu][1].split(','):
             if name != '':
                        datatmp.append(Idata[nu][0])
                        datatmp.append(name.strip())
                        datatmp.append(Idata[nu][2])
                        datatmp.append(Idata[nu][3])
                        datatmp.append(Idata[nu][4])
                        datatmp.append(Idata[nu][5])
                        data.append(datatmp)
                        datatmp = []
data1 = pd.DataFrame(data)
data1.columns = ['titel','name','date','vender','index','indexc']
data1.to_csv('./data/pdata.csv',sep='\t',index=False)

# --------------------
data = []
datatmp = []
Idata = get_data1()
for nu in range(len(Idata)):
        if nu % 10000 == 0:
                print ('finish -- %d / %d') %(nu,len(Idata))
        if len(Idata[nu][5].split(',')) > 1:
                for name in Idata[nu][5].split(','):
                    if name != '':
                        datatmp.append(Idata[nu][0])
                        datatmp.append(Idata[nu][1])
                        datatmp.append(Idata[nu][2])
                        datatmp.append(Idata[nu][3])
                        datatmp.append(Idata[nu][4])
                        datatmp.append(name)
                        data.append(datatmp)
                        datatmp = []
        else:
                datatmp.append(Idata[nu][0])
                datatmp.append(Idata[nu][1])
                datatmp.append(Idata[nu][2])
                datatmp.append(Idata[nu][3])
                datatmp.append(Idata[nu][4])
                datatmp.append('')
                data.append(datatmp)
                datatmp = []
data2 = pd.DataFrame(data)
data2.columns = ['titel','name','date','vender','index','indexc']
data2.to_csv('./data/dataO.csv',sep='\t',index=False)

