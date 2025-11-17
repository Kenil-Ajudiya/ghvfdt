import psrchive
import numpy as np

#pfds=glob.glob('pfd_select/*.pfd')

def nosigbins(tp):#finding the off pulse region using mean/rms ratio
	bins=tp.shape[1]
	offbins=[]
	for i in range(bins):
		d=tp[:,i]
		mnrms=np.mean(d)/np.std(d)
		if(mnrms<0.3): # threshold for finding off-pulse region... change as required
			offbins.append(i)
	return offbins

def pfd_data(pfd):
	var=psrchive.Archive_load(pfd)
	var.dedisperse()
	var.centre_max_bin()
	var.remove_baseline()
	return var.get_data()

def nullsubint(tp): #works for any 2d plot
	offbins=nosigbins(tp)
	subints=tp.shape[0]
	for k in range(1,subints//2+1,1):
		snrints=[]
		for i in range(0,subints,k):
			mbin=min(i+k,subints)
			prof=tp[i:mbin,:]
			prof=np.mean(prof,axis=0)
			offreg=prof[offbins]
			snrints.append(np.max(prof)/np.std(offreg))
		if(max(snrints)<5):
			print("Signal not found with subints "+str(k)+". Using "+str(k+1)+" subints")
		else:
			break
	
	if(max(snrints)<5):
		print("Pulsar not detected in half sub-int range. Hence Nulling can not be detected")
		null1=[]
	else:
		null=[]
		for i in range(len(snrints)):
			if(snrints[i]<np.mean(snrints)/2): # Nulling detection threshold
				null.append(i)
		null=np.array(null)*k
		null1=[]
		for i in null:
			null1.append(i)
			for j in range(1,k):
				null1.append(i+j)
	
	null=np.copy(np.array(null1))
	return null

def calcsnr(tp):
	offbins=nosigbins(tp)
	profile=np.mean(tp,axis=0)
	offreg=profile[offbins]
	return np.max(profile)/np.std(offreg)

def nullsnr(tp):
	subints=tp.shape[0]
	null=nullsubint(tp)
	oppnull=[i for i in range(subints) if i not in null]
	nullsnr=calcsnr(tp[oppnull,:])
	return nullsnr
	
