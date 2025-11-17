import numpy as np
from nulling import calcsnr

def findchan(fp):
	chans=fp.shape[0]
	chanmean=[]
	for i in range(chans):
		chanmean.append(np.mean(fp[i,:]))
	mx2md=np.max(chanmean)/abs(np.median(chanmean))
	if(mx2md<50 or np.median(chanmean)):
		return []
	else:
		return chanmean.index(np.max(chanmean))


def snrcharm(fp):
	chans=fp.shape[0]
	null=findchan(fp)
	oppnull=[i for i in range(chans) if i not in null]
	nullsnr=calcsnr(fp[oppnull,:])
	return nullsnr
