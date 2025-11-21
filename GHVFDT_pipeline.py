import numpy as np
import psrchive
import subprocess
from pathlib import Path
from cyclopts import App, Parameter
from typing import Annotated
from PIL import Image
from riptide.pipelines import Candidate, CandidatePlot

app = App("GH-VFDT Pipeline",
          help="A pipeline to classify and filter pulsar candidates using GH-VFDT and various filters.",
          version="1.0")

def make_PDF(files, output_pdf="Positive_Candidates.pdf", file_type="hdf5"):
    """
    Make a single, multi-page PDF document of the candidate files.

    Arguments:
    ----------
    files : list
        List of file paths to include in the PDF.
    output_pdf : str
        Output PDF file name.
    file_type : str
        Type of candidate files in the data directory. Options are pfd or hdf5.
    -----------
    """

    files = [Path(file) for file in files]
    repo_path = Path.cwd()
    img_files: list[Path] = [repo_path / (file.name + ".png") for file in files]

    if file_type == "hdf5":
        for i in range(len(files)):
            print(f"Generating PNG file for {files[i]}...")
            candPlotObj = CandidatePlot(Candidate.load_hdf5(files[i]), figsize=(16, 5), dpi=80)
            candPlotObj.saveimg(img_files[i])
    else:
        for pfd in files:
            print(f"Generating PostScript for {pfd}...")
            subprocess.run(f"show_pfd -noxwin {pfd}", shell=True, stdout=subprocess.DEVNULL)

    # Collect images ensuring RGB mode (PDF pages must be RGB)
    images: list[Image.Image] = []
    for img_file in img_files:
        if not img_file.exists():
            print(f"Warning: image file missing and skipped: {img_file}")
            continue
        with Image.open(img_file) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            images.append(img.copy())  # copy so context manager can close

    if not images:
        print("No images available to write PDF.")
    elif len(images) == 1:
        images[0].save(output_pdf, format="PDF")
    else:
        images[0].save(output_pdf, format="PDF", save_all=True, append_images=images[1:])

    if file_type == "pfd":
        subprocess.run("rm -r *.pfd*", shell=True)
    else:
        for img_file in img_files:
            img_file.unlink()

def nosigbins(tp): # Finding the off pulse region using mean/rms ratio
	bins=tp.shape[1]
	offbins=[]
	for i in range(bins):
		d=tp[:,i]
		mnrms=np.mean(d)/np.std(d)
		if(mnrms<0.3): # Threshold for finding off-pulse region... change as required
			offbins.append(i)
	return offbins

def pfd_data(pfd):
	var=psrchive.Archive_load(pfd)
	var.dedisperse()
	var.centre_max_bin()
	var.remove_baseline()
	return var.get_data()

def nullsubint(tp): # Works for any 2D plot
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

@app.default
def main(
    data_dir: Annotated[str, Parameter(name=["--data_dir", "-p"], help="Path to the directory containing all the pfd files. It is required if scores are not provided.")] = "/lustre_archive/spotlight/Kenil/ghvfdt/data",
    file_type: Annotated[str, Parameter(name=["--file_type", "-f"], help="Type of candidate files in the data directory. Options are pfd or hdf5")] = "hdf5",
    scores_file: Annotated[Path, Parameter(name=["--scores_file", "-s"], help="Path to scores file. Scores are calculated and stored in scores.arff if not available.")] = Path(""),
    model: Annotated[Path, Parameter(name=["--model", "-m"], help="Path to the trained GH-VFDT model")] = "/lustre_archive/spotlight/Kenil/ghvfdt/GHRSS_models/GHRSS1-3.model",
    fp_nulling: Annotated[bool, Parameter(name=["--fp_nulling", "-fp"], help="Apply filter for nulling in frequency profile")] = False,
    tp_nulling: Annotated[bool, Parameter(name=["--tp_nulling", "-tp"], help="Apply filter for nulling in time profile")] = False,
    schan: Annotated[bool, Parameter(name=["--schan", "-sc"], help="Apply filter for strong channel removal")] = False,
):
    """
    Run GH-VFDT pipeline.
    
    Arguments:
        pfd_dir (str): Path to the directory containing all the pfd files. It is required if scores are not provided.
        scores_file (Path): Path to scores file. Scores are calculated and stored in scores.arff if not available.
        model (Path): Path to the trained GH-VFDT model.
        fp_nulling (bool): Apply filter for nulling in frequency profile.
        tp_nulling (bool): Apply filter for nulling in time profile.
        schan (bool): Apply filter for strong channel removal.
    """

    if not (file_type == "pfd" or file_type == "hdf5"):
        raise ValueError("Invalid file_type. Choose either 'pfd' or 'hdf5'.")

    repo_path=Path.cwd()
    if not scores_file.is_file():
        scores_file = repo_path / "scores.arff"
        if scores_file.is_file():
            scores_file.unlink()
        subprocess.run(f"python CandidateScoreGenerators/ScoreGenerator.py -c {data_dir} -o {scores_file} --{file_type} --arff --dmprof", shell=True)

    predict_path = repo_path / "predict.txt"
    if predict_path.is_file():
        predict_path.unlink()
    predict_neg_path = repo_path / "predict.txt.negative"
    if predict_neg_path.is_file():
        predict_neg_path.unlink()
    print("\nGHVFDT Prediction Command:")
    print(f"java -jar ML.jar -v -m{model} -o{predict_path} -p{scores_file} -a1\n") # No space between the option flags and their values.
    subprocess.run(f"java -jar ML.jar -v -m{model} -o{predict_path} -p{scores_file} -a1", shell=True)
    
    scores_file.unlink()
    
    # Read GH-VFDT predict.txt and make a PDF of all positive candidates.
    with open(predict_path) as f:
        files = [line.split(",")[0] for line in f][1:]

    if file_type == "pfd":
        make_PDF(files, output_pdf="Positive_Candidates.pdf", file_type=file_type)

        if (fp_nulling or tp_nulling or schan):
            # Apply filters
            pfd_dat, null_fp, null_tp, schan_rm = [pfd_data(pfd) for pfd in files], [], [], []
            for j in range(len(pfd_dat)):
                i=pfd_dat[j]
                fp=np.mean(i[:,0,:],axis=0)
                chans=fp.shape[0]
                fp=fp[int(chans*0.1):-int(chans*0.1)]
                tp=np.mean(i[:,0,:],axis=1)
                snrfp=nullsnr(fp)
                snrtp=nullsnr(tp)
                snrschan=snrcharm(fp)
                if(snrfp>4.0): # Threshold for fp nulling removed SNR
                    null_fp.append(files[j])
                if(snrtp>4.0): # Threshold for tp nulling removed SNR
                    null_tp.append(files[j])
                if(snrschan>4.0): # Threshold for Strong channel removed candidates SNR
                    schan_rm.append(files[j])

            filtered=[]
            if(schan==False):
                if(fp_nulling==True and tp_nulling==False):
                    filtered=null_fp
                elif(fp_nulling==False and tp_nulling==True):
                    filtered=null_tp
                elif(fp_nulling==True and tp_nulling==True):
                    for i in null_fp:
                        if i in null_tp:
                            filtered.append(i)
            else:
                if(fp_nulling==True and tp_nulling==False):
                    for i in schan_rm:
                        if i in null_fp:
                            filtered.append(i)
                elif(fp_nulling==False and tp_nulling==True):
                    for i in schan_rm:
                        if i in null_tp:
                            filtered.append(i)
                elif(fp_nulling==True and tp_nulling==True):
                    for i in null_fp:
                        if i in null_tp:
                            if i in schan_rm:
                                filtered.append(i)

            make_PDF(filtered, output_pdf="Filtered_Candidates.pdf", file_type=file_type)
    else:
        make_PDF(files, output_pdf="Positive_Candidates.pdf", file_type=file_type)

if __name__ == "__main__":
    app()