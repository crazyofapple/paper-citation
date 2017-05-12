# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np

data  = pd.read_csv('./data/dataO.csv',sep='\t')
data['label'] = 1
data1 = pd.read_csv('./data/pdata.csv',sep='\t')
data1['label'] = 1
authors = pd.read_csv('./data/author.txt',header=None,sep='\t',names=['id','name'])

def get_authorByauthor():
	authindex = data1.groupby(['name','index'],as_index=False).sum()[['name','index']]
	authindexc = data.groupby(['name','indexc'],as_index=False).sum()[['name','indexc']]
	authHindex = get_CitaM3()[['name','hindex']]
	authbyauth = pd.merge(authindex,authindexc,how='left',left_on='index',right_on='indexc')
	authbyauth = pd.merge(authbyauth,authHindex,how='left',left_on='name_y',right_on='name')
	authbyauth = authbyauth.fillna(0)
	authbyauthtest = np.array(authbyauth)
        I = []
        Itmp = []
        pre = authbyauthtest[0][0]
        a = []
        for nu in range(len(authbyauthtest)):
                if nu % 10000 ==0:
                        print 'authbyauthtest number (min mean max,sum) finish -- %d / %d' %(nu,len(authbyauthtest))
                if authbyauthtest[nu][0]  == pre:
                        a.append(authbyauthtest[nu][5])
                else:
                        a = np.array(a)
                        Itmp.append(authbyauthtest[nu-1][0])
                        Itmp.append(a.mean())
                        Itmp.append(a.max())
                        Itmp.append(a.sum())
                        I.append(Itmp)
                        Itmp = []
                        a = []
                        a.append(authbyauthtest[nu][5])
                        pre = authbyauthtest[nu][0]
        result = pd.DataFrame(I)
        result.columns = ['name','citation_mean','citation_max','citation_sum']
        result = result.fillna(0)
	return result
	

# ----------- total paper for every author ----------------
def get_authorToNum(year):
	au = data1[(data1.date == year)]
	return au.groupby(['name'],as_index=False).sum()[['name','label']]

# ---------- get citation for every author paper all years  ----------------
def get_CAll(year):
	authorC = data[data.date == year][['name','label']]
	return authorC.groupby(['name'],as_index=False).sum()

# ----------  get train data ---------------------------------------
def get_trainId():
	train = pd.read_csv('./data/citation_train.txt',sep='\t',header=None)
	train.columns=['nameid','name','result']
	train_2012 = get_AuthorCitation(2012)
	train = pd.merge(train,train_2012,how='left',on='name')
	train['delte'] = train['result'] - train['label']
	#trainClass1 = train[train.delte > 1000][['nameid','name','result']]
	#trainClass1['class'] = 1
	#trainClass2 = train[train.delte < 1000][['nameid','name','result']]
        #trainClass2['class'] = 0
	#train.columns=['nameid','name','result']
	return train[['nameid','name','result']]

def get_testId():
	test = pd.read_csv('./data/citation_test.txt',sep='\t',header=None)
	test.columns=['nameid','name']
	return test

# --------------- get co-author ----------------------------
def get_coauthor(start_time=1935,end_time=2011):
	indnu = data1[(data1.date >= start_time) & (data1.date <= end_time)].groupby(['index','name'],as_index=False).sum()[['index','name','label']]
	auind = data1[(data1.date >= start_time) & (data1.date <= end_time)].groupby(['name','index'],as_index=False).sum()[['name','index']]
	authnu = pd.merge(auind,indnu,how='left',on='index')
	authHindex = get_CitaM3(start = start_time,end = end_time)[['name','hindex']]
	authnuHindex = pd.merge(authnu,authHindex,how='left',left_on='name_y',right_on='name')[['name_x','hindex']]
	authnuHindex = authnuHindex.fillna(0)	
	authnuHindextest = np.array(authnuHindex)
	I = []
	Itmp = []
	pre = authnuHindextest[0][0]
	a = []
	for nu in range(len(authnuHindextest)):
		if nu % 10000 ==0:
			print 'Citation number (min mean max,sum) finish -- %d / %d' %(nu,len(authnuHindextest))
		if authnuHindextest[nu][0]  == pre:
			a.append(authnuHindextest[nu][1])
		else:
			a = np.array(a)
			Itmp.append(authnuHindextest[nu-1][0])
			Itmp.append(a.min())
			Itmp.append(a.mean())
			Itmp.append(a.max())
			Itmp.append(a.sum())
			I.append(Itmp)
			Itmp = []
			a = []
			a.append(authnuHindextest[nu][1])
			pre = authnuHindextest[nu][0]
	authm3 = pd.DataFrame(I)
	authm3.columns = ['name','citation_min','citation_mean','citation_max','citation_sum']
	authm3 = authm3.fillna(0)

	authnu1 =  authnu.groupby('name_x',as_index=False).sum()
	authnu1.columns=['name','label']
	authnu2 = data1[(data1.date >= start_time) & (data1.date <= end_time)].groupby(['name'],as_index=False).sum()[['name','label']]
	authnu1['labelre'] = authnu1['label'] - authnu2['label']
	results = authnu1[['name','labelre']]
	results = pd.merge(authnu1,authm3,how='left',on='name')
	results['citation_sum'] = (results['citation_sum'] - results['citation_sum'].min() ) / (results['citation_sum'].max() - results['citation_sum'].min())
	return results
		

# --------- get total number of vender for each authors ------------
def get_num_vender():
	num_vend = data1.groupby(['name','vender'],as_index=False).sum()[['name','vender']]
	num_vend['label'] = 1
	return num_vend.groupby('name',as_index=False).sum()
# ----------------- 出版社被引用 文章次数的hindex 等相关统计 --------------
def get_venderOindex(start,end):
	vendnu = data1.groupby('vender',as_index=False).sum()[['vender','label' ]]
	paper_ci = data[(data.date >= start) & (data.date <= end)].groupby('indexc',as_index=False).sum()[['indexc','label']]
	vendata = data.groupby(['vender','index'],as_index=False).sum()[['vender','index']]
	vend = pd.merge(vendata,paper_ci,how='left',left_on='index',right_on='indexc')
	vend['label'] = vend['label'].fillna(0)
	vend = vend[['vender','index','label']]
	vend = vend.sort_values(by=['vender','label'],ascending=False)
	vendtest = np.array(vend)
        I = []
        Itmp = []
        pre = vendtest[0][0]
        a = []
	for nu in range(len(vendtest)):
                if nu % 10000 ==0:
                        print ' vender hindex finish -- %d / %d' %(nu,len(vendtest))
                if (vendtest[nu][0]  == pre) and (nu != len(vendtest)):
                        a.append(vendtest[nu][2])
                else:
                        a = np.array(a)
                        Itmp.append(vendtest[nu-1][0])
                        Itmp.append(a.min())
                        Itmp.append(a.mean())
			Itmp.append(a.max())
                        Itmp.append(a.sum())
                        for  i in range(len(a)):
                                flag = 0
                                if i >= a[i]:
                                        Itmp.append(i)
                                        flag = 1
                                        break
                        if flag == 0 :
                                Itmp.append(len(a))
                        I.append(Itmp)
                        Itmp = []
                        a = []
                        a.append(vendtest[nu][2])
                        pre = vendtest[nu][0]
        vend = pd.DataFrame(I)
        vend.columns = ['vender','vend_min','vend_mean','vend_max','vend_sum','vend_hindex']
        vend = vend.fillna(0)
	result = pd.merge(vend,vendnu,how='left',on='vender')
	result['vend_sum'] = result['vend_sum'] / result['label']
	result = result.fillna(0)
        return result

# -------------根据 期刊被引用数 统计特征 --------------------------------
def get_fea_authors_venderO(start=1935,end=2012):
	vend = get_venderOindex(start,end)
	vend.columns = ['vender','vend_min','vend_mean','vend_max','vend_sum','vend_hindex','perpaper']
	vend['perpaper'] = (vend['perpaper'] - vend['perpaper'].min()) / (vend['perpaper'].max() - vend['perpaper'].min())
	audata = data1.groupby(['name','vender','index'],as_index=False).sum()
	del audata['date']
	audata = pd.merge(audata,vend,how='left',on='vender')
	del audata['vender']
	del audata['index']
	del audata['label']
	audata = audata.fillna(1)
	audatatest = np.array(audata)
	I = []
	Itmp = []
	pre = audatatest[0][0]
	amin = []
	amean = []
	amax = []
	asum = []
	ahin = []
	apap = []
	for nu in range(len(audatatest)):
		if nu % 10000 ==0:
			print 'vend hindex  finish -- %d / %d' %(nu,len(audatatest))
		if audatatest[nu][0]  == pre:
			amin.append(audatatest[nu][1])
			amean.append(audatatest[nu][2])
			amax.append(audatatest[nu][3])
			asum.append(audatatest[nu][4])
			ahin.append(audatatest[nu][5])
			apap.append(audatatest[nu][6])
		else:
			amin = np.array(amin)
			amean = np.array(amean)
			amax = np.array(amax)
			asum = np.array(asum)
			ahin = np.array(ahin)
			apap = np.array(apap)
			Itmp.append(audatatest[nu-1][0])
			Itmp.append(amin.min())
			Itmp.append(amin.mean())
			Itmp.append(amin.max())
			Itmp.append(amin.sum())
			Itmp.append(amean.min())
                        Itmp.append(amean.mean())
                        Itmp.append(amean.max())
                        Itmp.append(amean.sum())
			Itmp.append(amax.min())
                        Itmp.append(amax.mean())
                        Itmp.append(amax.max())
                        Itmp.append(amax.sum())
                        Itmp.append(asum.min())
                        Itmp.append(asum.mean())
                        Itmp.append(asum.max())
                        Itmp.append(asum.sum())
			Itmp.append(ahin.min())
                        Itmp.append(ahin.mean())
                        Itmp.append(ahin.max())
                        Itmp.append(ahin.sum())
                        Itmp.append(apap.min())
                        Itmp.append(apap.mean())
                        Itmp.append(apap.max())
                        Itmp.append(apap.sum())
			I.append(Itmp)
			Itmp = []
			amin = []
			amean = []
			amax = []
			asum = []
			ahin = []
			apap = []
			amin.append(audatatest[nu][1])
                        amean.append(audatatest[nu][2])
                        amax.append(audatatest[nu][3])
                        asum.append(audatatest[nu][4])
                        ahin.append(audatatest[nu][5])
			apap.append(audatatest[nu][6])
			pre = audatatest[nu][0]
	authVend = pd.DataFrame(I)
	authVend.columns = ['name','vend_min_min','vend_min_mean','vend_min_max','vend_min_sum','vend_mean_min','vend_mean_mean','vend_mean_max','vend_mean_sum','vend_max_min','vend_max_mean','vend_max_max','vend_max_sum','vend_sum_min','vend_sum_mean','vend_sum_max','vend_sum_sum','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum','vend_pap_min','vend_pap_mean','vend_pap_max','vend_pap_sum']
	authVend = authVend.fillna(0)
	del authVend['vend_min_sum']
	del authVend['vend_mean_sum']
	del authVend['vend_max_sum']
	del authVend['vend_pap_min']
        del authVend['vend_pap_mean']
        del authVend['vend_pap_max']
        del authVend['vend_pap_sum']
        del authVend['vend_sum_min']
        del authVend['vend_sum_mean']
        del authVend['vend_sum_max']
        del authVend['vend_sum_sum']	
	#del authVend['vend_min_sum']
	return authVend
	

# ---------- 出版社引用别人文章的总数 ------------------------
def get_venderHindex(start,end):
	vend = data[(data.date >= start) &(data.date <= end) ].groupby(['vender','indexc'],as_index=False).sum()[['vender','indexc','label']]
	vend = vend.sort_values(by=['vender','label'],ascending=False)
	vendtest = np.array(vend)
	I = []
	Itmp = []
	pre = vendtest[0][0]
	a = []
	for nu in range(len(vendtest)):
		if nu % 10000 ==0:
			print ' vender hindex finish -- %d / %d' %(nu,len(vendtest))
		if (vendtest[nu][0]  == pre) and (nu != len(vendtest)):
			a.append(vendtest[nu][2])
		else:
			a = np.array(a)
			Itmp.append(vendtest[nu-1][0])
			Itmp.append(a.min())
			Itmp.append(a.mean())
			Itmp.append(a.sum())
			for  i in range(len(a)):
				flag = 0
				if i >= a[i]:
					Itmp.append(i)
					flag = 1
					break
			if flag == 0 :
				Itmp.append(len(a))
			I.append(Itmp)
			Itmp = []
			a = []
			a.append(vendtest[nu][2])
			pre = vendtest[nu][0]
	vend = pd.DataFrame(I)
	vend.columns = ['vender','vend_min','vend_mean','vend_max','vend_hindex']
	vend = vend.fillna(0)
	return vend

def get_fea_authors_vender(start=1935,end=2012):
	vend = get_venderHindex(start,end)
	audata = data1.groupby(['name','vender','index'],as_index=False).sum()
	del audata['date']
	audata = pd.merge(audata,vend,how='left',on='vender')
	del audata['vender']
	del audata['index']
	del audata['label']
	audata = audata.fillna(1)
	audatatest = np.array(audata)
	I = []
	Itmp = []
	pre = audatatest[0][0]
	amin = []
	amean = []
	amax = []
	ahin = []
	for nu in range(len(audatatest)):
		if nu % 10000 ==0:
			print 'vend hindex  finish -- %d / %d' %(nu,len(audatatest))
		if audatatest[nu][0]  == pre:
			amin.append(audatatest[nu][1])
			amean.append(audatatest[nu][2])
			amax.append(audatatest[nu][3])
			ahin.append(audatatest[nu][4])
		else:
			amin = np.array(amin)
			amean = np.array(amean)
			amax = np.array(amax)
			ahin = np.array(ahin)
			Itmp.append(audatatest[nu-1][0])
			Itmp.append(amin.min())
			Itmp.append(amin.mean())
			Itmp.append(amin.max())
			Itmp.append(amin.sum())
			Itmp.append(amean.min())
                        Itmp.append(amean.mean())
                        Itmp.append(amean.max())
                        Itmp.append(amean.sum())
			Itmp.append(amax.min())
                        Itmp.append(amax.mean())
                        Itmp.append(amax.max())
                        Itmp.append(amax.sum())
			Itmp.append(ahin.min())
                        Itmp.append(ahin.mean())
                        Itmp.append(ahin.max())
                        Itmp.append(ahin.sum())
			I.append(Itmp)
			Itmp = []
			amin = []
			amean = []
			amax = []
			ahin = []
			amin.append(audatatest[nu][1])
                        amean.append(audatatest[nu][2])
                        amax.append(audatatest[nu][3])
                        ahin.append(audatatest[nu][4])
			pre = audatatest[nu][0]
	authVend = pd.DataFrame(I)
	authVend.columns = ['name','vend_min_min','vend_min_mean','vend_min_max','vend_min_sum','vend_mean_min','vend_mean_mean','vend_mean_max','vend_mean_sum','vend_max_min','vend_max_mean','vend_max_max','vend_max_sum','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']
	authVend = authVend.fillna(0)
	del authVend['vend_min_sum']
	del authVend['vend_mean_sum']
	del authVend['vend_max_sum']
	#del authVend['vend_min_sum']
	return authVend
	
			
	
	

def get_age():
	dage = data1.groupby(['name','date'],as_index=False).sum()
	dagetest = np.array(dage.fillna(0))
	I = []
        Itmp = []
        pre = dagetest[0][0]
        a = []
	b= []
	for nu in range(len(dagetest)):
		if nu % 10000 ==0:
			print 'Authors age  finish -- %d / %d' %(nu,len(dagetest))
		if dagetest[nu][0]  == pre:
			a.append(dagetest[nu][1])
			b.append(dagetest[nu][2])
		else:
			a = np.array(a,dtype=int)
			b = np.array(b,dtype=int)
			Itmp.append(dagetest[nu-1][0])
			if a.min() != 2017:
				Itmp.append((2012 - a.min()))
				Itmp.append((b.sum() *1.0/ (2012 - a.min())))
			else:
				Itmp.append((2012 - 1936))
				Itmp.append((b.sum()*1.0 / 75))
			I.append(Itmp)
			Itmp = []
			a = []
			b = []
			a.append(dagetest[nu][1])
			b.append(dagetest[nu][2])
			pre = dagetest[nu][0]
        age = pd.DataFrame(I)
        age.columns = ['name','authors_age','papernumPerYear']
        age = age.fillna(0)
        return age
	
def get_CitaM3(start=1936,end=2011):
	m3 = data[(data.date <= end) & (data.date >= start)].groupby(['indexc'],as_index=False).sum()[['indexc','label']]
	nameIndex = data1[['name','index']]
	cin  = pd.merge(nameIndex,m3,left_on='index',right_on='indexc',how='left')
	cin = cin.fillna(0)
	cin = cin.sort_values(by=['name','label'],ascending=False)
	cintest = np.array(cin)
	I = []
	Itmp = []
	pre = cintest[0][0]
	a = []
	for nu in range(len(cintest)):
		if nu % 10000 ==0:
			print 'Citation number (min mean max,sum,hindex) finish -- %d / %d' %(nu,len(cintest))
		if cintest[nu][0]  == pre:
			a.append(cintest[nu][3])
		else:
			a = np.array(a)
			Itmp.append(cintest[nu-1][0])
			Itmp.append(a.min())
			Itmp.append(a.mean())
			Itmp.append(a.max())
			Itmp.append(a.sum())
			for  i in range(len(a)):
				flag = 0
				if i >= a[i]:
					Itmp.append(i)
					flag = 1
					break
			if flag == 0:
				Itmp.append(len(a))
			I.append(Itmp)
			Itmp = []
			a = []
			a.append(cintest[nu][3])
			pre = cintest[nu][0]

	citam3 = pd.DataFrame(I)
	citam3.columns = ['name','citation_min','citation_mean','citation_max','citation_sum','hindex']
	citam3 = citam3.fillna(0)
	return citam3
		

def get_Citation(start,end):
	return data[(data.date <= end) & (data.date > start)].groupby(['name'],as_index=False).sum()[['name','label']]
	
def get_AuthorCitation(end):
	paperbycition = data[data.date <= end].groupby(['indexc'],as_index=False).sum()[['indexc','label']]
	authindex = data1.groupby(['name','index'],as_index=False).sum()[['name','index']]
	authcita = pd.merge(authindex,paperbycition,how='left',left_on='index',right_on='indexc')
	authcita = authcita.fillna(0)
	return authcita.groupby('name',as_index=False).sum()

def get_AuthorCitationdelte(start,end):
        paperbycition = data[(data.date <= end) & (data.date >= start)].groupby(['indexc'],as_index=False).sum()[['indexc','label']]
        authindex = data1.groupby(['name','index'],as_index=False).sum()[['name','index']]
        authcita = pd.merge(authindex,paperbycition,how='left',left_on='index',right_on='indexc')
        authcita = authcita.fillna(0)
        return authcita.groupby('name',as_index=False).sum()




# ---------- get citation for every paper in 2011 -------------------
def get_train():
	train_end_year = 2011
	authorToNum = data1.groupby(['name'],as_index=False).sum()[['name','label']]
	authorCNum = data.groupby(['name'],as_index=False).sum()[['name','label']]

	for year in range(5):
		train_year = train_end_year - year
		authorToNum = pd.merge(authorToNum,get_authorToNum(train_year),how='left',on='name') 
		authorCNum = pd.merge(authorCNum,get_CAll(train_year),how='left',on='name') 
	train = get_trainId()	

	#train_end_year = 2011
	train = pd.merge(train,authorCNum,how='left',on='name')
	train = pd.merge(train,authorToNum,how='left',on='name')

	for year in (1,2,3,4,5,10,20,40,60):
		train_start_year = 2011 - year
		train = pd.merge(train,get_Citation(start=train_start_year,end=train_end_year),how='left',on='name')

	
	train = pd.merge(train,get_CitaM3(),how='left',on='name')

	# ---- add hindex delte in 2 years -------------------------
	train1 = get_CitaM3()
	train2 = get_CitaM3(1936,2009)
	train3 = train1[['citation_min','citation_mean','citation_max','citation_sum','hindex']] - train2[['citation_min','citation_mean','citation_max','citation_sum','hindex']]
	train3['name'] = train1['name']
	train = pd.merge(train,train3,how='left',on='name')
	
	# ---- add authors ages and num paper per year ---------
	train = pd.merge(train,get_age(),how='left',on='name')
	
	# ----- add author vender ------------------
	train = pd.merge(train,get_fea_authors_vender(),how='left',on='name')
	
	# -------- add vender hindex delte ------------
	train1 = get_fea_authors_vender()
	train2 = get_fea_authors_vender(start=1935,end=2009)
	train3 = train1[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']] - train2[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']]
	train3['name'] = train1['name']
	train = pd.merge(train,train3,how='left',on='name')
	
	# --------- add num vender per authors --------
	train = pd.merge(train,get_num_vender(),how='left',on='name')
	# --------- add co author --------------------
	for year in range(1):
		train_start_year = 2011 - year
		train1 = get_coauthor(start_time=1935,end_time=train_start_year)
		train2 = get_coauthor(start_time=1935,end_time=train_start_year - 3)
		train3 = pd.merge(train1,train2,how='left',on='name')
		train3['label_delte'] = train3['label_x'] - train3['label_y']
		train3['labelre_delte'] = train3['labelre_x'] - train3['labelre_y']
		train3['citation_min_delte'] = train3['citation_min_x'] - train3['citation_min_y']
		train3['citation_mean_delte'] = train3['citation_mean_x'] - train3['citation_mean_y']
		train3['citation_max_delte'] = train3['citation_max_x'] - train3['citation_max_y']
		train3['citation_sum_delte'] = train3['citation_sum_x'] - train3['citation_sum_y']

	train = pd.merge(train,train3,how='left',on='name')

	# -------- add 被引用次数 for per authors -----------
        train1 = get_fea_authors_venderO()
        train2 = get_fea_authors_venderO(start=1935,end=2010)
        train3 = train1[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_max_max','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']] - train2[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_max_max','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']]
        train3['name'] = train1['name']
        train = pd.merge(train,train3,how='left',on='name')
	train = pd.merge(train,train1,how='left',on='name')
	# -------add auth by auth -----------
	train = pd.merge(train,get_authorByauthor(),how='left',on='name')
	
	
	for year in range(3):
		train_start_year = 2011 - year
		train = pd.merge(train,get_CitaM3(start=train_start_year,end=train_end_year),how='left',on='name')
		train = pd.merge(train,get_CitaM3(end=train_start_year),how='left',on='name')
	
	train = train.fillna(0)
	authors = train[['nameid','name']].copy()
	result = train['result'].copy()
	del train['nameid']
	del train['name']
	del train['result']
	return authors ,train,result

	
def get_test():
        #authorToNum = get_authorToNum()
	test_end_year = 2011
        authorToNum = data1.groupby(['name'],as_index=False).sum()[['name','label']]
	authorCNum = data.groupby(['name'],as_index=False).sum()[['name','label']]
        for year in range(5):
                test_year = test_end_year - year
                authorToNum = pd.merge(authorToNum,get_authorToNum(test_year),how='left',on='name')
		authorCNum = pd.merge(authorCNum,get_CAll(test_year),how='left',on='name')
	test = get_testId()
        #test_end_year = 2011
        test = pd.merge(test,authorCNum,how='left',on='name')
        test = pd.merge(test,authorToNum,how='left',on='name')
        for year in (1,2,3,4,5,10,20,40,60):
                test_start_year = 2011 - year
                test = pd.merge(test,get_Citation(start = test_start_year,end = test_end_year),how='left',on='name')

        test = pd.merge(test,get_CitaM3(),how='left',on='name')
	# ---- add hindex delte in 2 years -------------------------
        test1 = get_CitaM3()
        test2 = get_CitaM3(1936,2009)
        test3 = test1[['citation_min','citation_mean','citation_max','citation_sum','hindex']] - test2[['citation_min','citation_mean','citation_max','citation_sum','hindex']]
        test3['name'] = test1['name']
        test = pd.merge(test,test3,how='left',on='name')
	# ---- add authors ages and num paper per year ---------
	test = pd.merge(test,get_age(),how='left',on='name')
	#  ----- add author vender -------------
	test = pd.merge(test,get_fea_authors_vender(),how='left',on='name')
	# -------- add vender hindex delte ------------
        test1 = get_fea_authors_vender()
        test2 = get_fea_authors_vender(start=1935,end=2010)
        test3 = test1[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']] - test2[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']]
        test3['name'] = test1['name']
        test = pd.merge(test,test3,how='left',on='name')
	
	# --------- add num vender per authors --------
        test = pd.merge(test,get_num_vender(),how='left',on='name')
	# -------- add co author ---------------------
        for year in range(1):
                test_start_year = 2011 - year
                test1 = get_coauthor(start_time=1935,end_time=test_start_year)
                test2 = get_coauthor(start_time=1935,end_time=test_start_year - 3)
                test3 = pd.merge(test1,test2,how='left',on='name')
                test3['label_delte'] = test3['label_x'] - test3['label_y']
                test3['labelre_delte'] = test3['labelre_x'] - test3['labelre_y']
                test3['citation_min_delte'] = test3['citation_min_x'] - test3['citation_min_y']
                test3['citation_mean_delte'] = test3['citation_mean_x'] - test3['citation_mean_y']
                test3['citation_max_delte'] = test3['citation_max_x'] - test3['citation_max_y']
                test3['citation_sum_delte'] = test3['citation_sum_x'] - test3['citation_sum_y']

        test = pd.merge(test,test3,how='left',on='name')
        
	
	# -------- add 被引用次数 for per authors -----------
	test1 = get_fea_authors_venderO()
        test2 = get_fea_authors_venderO(start=1935,end=2009)
        test3 = test1[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_max_max','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']] - test2[['vend_min_min','vend_min_mean','vend_min_max','vend_mean_min','vend_mean_mean','vend_mean_max','vend_max_min','vend_max_mean','vend_max_max','vend_hin_min','vend_hin_mean','vend_hin_max','vend_hin_sum']]
        test3['name'] = test1['name']
        test = pd.merge(test,test3,how='left',on='name')
	test = pd.merge(test,test1,how='left',on='name')
	# -------add auth by auth -----------
        test = pd.merge(test,get_authorByauthor(),how='left',on='name')
	
        for year in range(3):
                test_start_year = 2011 - year
                test = pd.merge(test,get_CitaM3(start = test_start_year,end = test_end_year),how='left',on='name')
                test = pd.merge(test,get_CitaM3(end = test_start_year),how='left',on='name')

	test = test.fillna(0)
	authors = test[['nameid','name']].copy()
        del test['nameid']
        del test['name']
	return authors ,test

if __name__ == '__main__':
    #user_index, training_data, label = get_train()
    #user_index.to_csv('./cache/user_index.csv',sep='\t')
    #training_data.to_csv('./cache/training_data.csv',sep='\t')
    #label.to_csv('./cache/label.csv',sep='\t')
    sub_user_index, sub_trainning_data = get_test()
    sub_user_index.to_csv('./cache/sub_user_index.csv',sep='\t')
    sub_trainning_data.to_csv('./cache/sub_trainning_data.csv',sep='\t')
	

		
		
	



