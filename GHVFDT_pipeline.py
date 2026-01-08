import numpy as np
from pandas import read_csv
import psrchive
import subprocess
import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path
from cyclopts import App, Parameter
from typing import Annotated
from PIL import Image
from riptide.pipelines import Candidate, CandidatePlot
from CandidateScoreGenerators.Utilities import MultiColorFormatter

app = App("GH-VFDT Pipeline",
          help="A pipeline to classify and filter pulsar candidates using GH-VFDT and various filters.",
          version="1.0")

def _render_hdf5_png(hdf5_file: Path):
    logger = logging.getLogger("ML_classifier")
    logger.info(f"Generating PNG for candidate {hdf5_file}")
    img_file = hdf5_file.with_suffix(".png")
    candPlotObj = CandidatePlot(Candidate.load_hdf5(hdf5_file), figsize=(16, 5), dpi=80)
    candPlotObj.saveimg(img_file)

def prep_data_dir(cand_csv: Path, all_cands_dir: Path) -> (Path | str):
    """
    Prepare the data directory by copying candidates listed in the CSV file.

    Arguments:
    ----------
    cand_csv : Path
        Path to the candidate CSV file.
    all_cands_dir : Path
        Path to the directory to store candidate files.

    Returns:
    --------
    filtered_candies_dir : Path
        Path to the directory containing all the candidate files.
    file_type : str
        Type of candidate files in the data directory. Options are 'pfd' or 'hdf5'.
    """

    filtered_candies_dir = all_cands_dir / "Filtered_Candidates"
    filtered_candies_dir.mkdir(exist_ok=True)

    filtered_candies = read_csv(cand_csv)
    suffix = Path(filtered_candies["fname"][0]).suffix
    if suffix == ".h5":
        file_type = "hdf5"
    elif suffix == ".pfd":
        file_type = "pfd"
    else:
        raise ValueError(f"Invalid file suffix {suffix}. It should either either be '.h5' or '.pfd'.")

    i = 0
    filtered_cands = []
    for cand_file in filtered_candies["fname"]:
        cand = (all_cands_dir / f"BM{filtered_candies['beam'][i]}.down_RFI_Mitigated_01/candidates") / cand_file
        filtered_cands.append(cand)
        subprocess.run(f"cp {cand} {filtered_candies_dir}/BM{filtered_candies['beam'][i]}_{cand_file}", shell=True)
        i += 1
    return filtered_candies_dir, file_type

def make_PDF(files, output_pdf=Path("Positive_Candidates.pdf"), file_type="hdf5", njobs=None):
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
    img_files: list[Path] = [file.with_suffix(".png") for file in files]

    logger = logging.getLogger("ML_classifier")
    if file_type == "hdf5":
        with Pool(processes=njobs or cpu_count()) as pool:
            for _ in pool.imap_unordered(_render_hdf5_png, files):
                pass
    else:
        for pfd in files:
            logger.info(f"Generating PostScript for {pfd}")
            subprocess.run(f"show_pfd -noxwin {pfd}", shell=True, stdout=subprocess.DEVNULL)

    # Collect images ensuring RGB mode (PDF pages must be RGB)
    images: list[Image.Image] = []
    for img_file in img_files:
        if not img_file.exists():
            logger.warning(f"Image file missing; skipped {img_file}")
            continue
        with Image.open(img_file) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            images.append(img.copy())  # copy so context manager can close

    if not images:
        logger.warning("No images available to write PDF")
    elif len(images) == 1:
        images[0].save(output_pdf, format="PDF", quality=100)
        logger.info(f"Wrote single-page PDF to {output_pdf}")
    else:
        images[0].save(output_pdf, format="PDF", save_all=True, quality=100, append_images=images[1:])
        logger.info(f"Wrote {len(images)} pages to PDF {output_pdf}")

    if file_type == "pfd":
        subprocess.run("rm -r *.pfd*", shell=True)

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
    logger = logging.getLogger("ML_classifier")
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
            logger.info(f"Signal not found with subints {k}. Trying {k+1} subints")
        else:
            break
	
    if(max(snrints)<5):
        logger.warning("Pulsar not detected in half sub-int range; nulling cannot be detected")
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

def configure_logger(name: str = "ML_classifier", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(MultiColorFormatter())
        logger.addHandler(stream_handler)
    return logger

@app.default
def main(
    cand_csv: Annotated[Path | None, Parameter(name=["--cand-csv", "-c"], help="Path to the candidate CSV file. If provided, only those candidates will be processed.")] = None,
    data_dir: Annotated[Path | None, Parameter(name=["--data-dir", "-d"], help="Path to the directory containing all the candidate files. It is required if scores are not provided.")] = None,
    file_type: Annotated[str, Parameter(name=["--file-type", "-f"], help="Type of candidate files in the data directory. Options are pfd or hdf5")] = "hdf5",
    output_pdf: Annotated[Path, Parameter(name=["--output-pdf", "-o"], help="Path to output PDF file containing positive candidates.")] = Path("Positive_Candidates.pdf"),
    scores_file: Annotated[Path, Parameter(name=["--scores-file", "-s"], help="Path to scores file. Scores are calculated and stored in scores.arff if not available.")] = Path(""),
    model: Annotated[Path, Parameter(name=["--model", "-m"], help="Path to the trained GH-VFDT model")] = "/lustre_archive/apps/tdsoft/ghvfdt/GHRSS_models/GHRSS1-3.model",
    fp_null: Annotated[bool, Parameter(name=["--fp-null", "-fp"], help="Apply filter for nulling in frequency profile")] = False,
    tp_null: Annotated[bool, Parameter(name=["--tp-null", "-tp"], help="Apply filter for nulling in time profile")] = False,
    schan: Annotated[bool, Parameter(name=["--schan", "-sc"], help="Apply filter for strong channel removal")] = False,
    njobs: Annotated[int | None, Parameter(name=["--njobs", "-j"], help="Number of parallel jobs to use for PDF generation. Defaults to number of CPU cores.")] = None,
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

    logger = configure_logger()
    logger.info("Starting the GH-VFDT pipeline.")

    if cand_csv:
        if not data_dir:
            data_dir = cand_csv.parent
        data_dir, file_type = prep_data_dir(cand_csv, data_dir) # data_dir is now a temporary directory.
        logger.info(f"Prepared temporary data directory {data_dir} with file type {file_type}")

    if not (file_type == "pfd" or file_type == "hdf5"):
        logger.error(f"Invalid file_type {file_type}. Expected 'pfd' or 'hdf5'")
        raise ValueError(f"Invalid file_type {file_type}. It should either either be 'pfd' or 'hdf5'.")

    temp_path=Path("/lustre_data/spotlight/data/watched")
    ghvfdt_path = "/lustre_archive/apps/tdsoft/ghvfdt"
    if not scores_file.is_file():
        scores_file = temp_path / "scores.arff"
        if scores_file.is_file():
            scores_file.unlink()
            logger.debug(f"Removed stale scores file {scores_file}")
        logger.info("Generating candidate scores (ARFF file)")
        subprocess.run(f"python {ghvfdt_path}/CandidateScoreGenerators/ScoreGenerator.py -c {data_dir} -o {scores_file} --{file_type} --arff --dmprof", shell=True)

    predict_path = output_pdf.parent / "predict.csv"
    if predict_path.is_file():
        predict_path.unlink()
    predict_neg_path = output_pdf.parent / "predict_negative.csv"
    if predict_neg_path.is_file():
        predict_neg_path.unlink()

    logger.info("Launching GH-VFDT ML classifier.")
    # No space between the option flags and their values.
    # -Djava.awt.headless=true to run java in headless mode since X11 forwarding is not required.
    subprocess.run(f"java -Djava.awt.headless=true -jar {ghvfdt_path}/ML.jar -v -m{model} -o{predict_path} -p{scores_file} -a1", shell=True)
    logger.info(f"Prediction completed; results at {predict_path}")

    try:
        scores_file.unlink()
        logger.debug(f"Removed temporary scores file {scores_file}")
    except Exception as e:
        logger.warning(f"Failed to remove scores file {scores_file}: {e}")

    # Read GH-VFDT predict.txt and make a PDF of all positive candidates.
    with open(predict_path) as f:
        files = [line.split(",")[0] for line in f][1:]

    if file_type == "hdf5":
        pass
        logger.info(f"Creating PDF for {len(files)} positive hdf5 candidates")
        make_PDF(files, output_pdf, file_type, njobs)
    else:
        logger.info(f"Creating PDF for {len(files)} positive pfd candidates")
        make_PDF(files, output_pdf, file_type, njobs)

        # Apply filters if selected.
        if (fp_null or tp_null or schan):
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
                if(fp_null==True and tp_null==False):
                    filtered=null_fp
                elif(fp_null==False and tp_null==True):
                    filtered=null_tp
                elif(fp_null==True and tp_null==True):
                    for i in null_fp:
                        if i in null_tp:
                            filtered.append(i)
            else:
                if(fp_null==True and tp_null==False):
                    for i in schan_rm:
                        if i in null_fp:
                            filtered.append(i)
                elif(fp_null==False and tp_null==True):
                    for i in schan_rm:
                        if i in null_tp:
                            filtered.append(i)
                elif(fp_null==True and tp_null==True):
                    for i in null_fp:
                        if i in null_tp:
                            if i in schan_rm:
                                filtered.append(i)

            logger.info(f"Filtered {len(filtered)} candidates after applying selected filters")
            make_PDF(filtered, f"{output_pdf.parent}/{output_pdf.stem}_Filtered.pdf", file_type, njobs)

    if cand_csv: # it was a temporary data directory.
        subprocess.run(f"rm -r {data_dir}", shell=True)
        logger.debug(f"Removed temporary data directory {data_dir}")

if __name__ == "__main__":
    app()