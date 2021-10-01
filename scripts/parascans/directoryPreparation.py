import os

def cleanDirectories(indir=""):
    os.system("rm -rf {0}/.submitit".format(indir))
    for root, dirs, files in os.walk(indir):
        
        if root.endswith("wandb"):
            os.chdir(root)
            os.system("mv debug.log ../.")
            os.chdir(indir)

        if "wandb" in dirs:
            os.chdir(root)
            os.system("rm -rf wandb")
            os.system("rm -rf .hydra")

            os.chdir(indir)
    return

def collectHistograms(indir=""):
    os.chdir(indir)
    COLLECTIONPATH=os.path.join(indir,"histogramCollection")
    os.system("mkdir -p {0}".format(COLLECTIONPATH))
    import glob
    EXAMPLEPATH=list(glob.iglob("0--*/histograms"))[0]
    for root, dirs, files in os.walk(os.path.join(indir,EXAMPLEPATH)):
        for f in files:
            os.system("mkdir -p {0}/{1}".format(COLLECTIONPATH,f.replace(".png","")))
        break

    for root, dirs, files in os.walk(indir):
        if "histograms" in root:
            os.chdir(root)
            runnumber=root.split("/")[-2]
            # .split("--")[0]

            for f in files:
                fname=f.replace(".png","")
                copyCMD="cp {0}.png {1}/{0}/{2}_{0}.png".format(fname,COLLECTIONPATH,runnumber)
                # print(copyCMD)
                # exit()
                os.system(copyCMD)

def collectPkl(indir=""):
    os.chdir(indir)
    COLLECTIONPATH=os.path.join(indir,"runSummaries")
    os.system("mkdir -p {0}".format(COLLECTIONPATH))
    for root, dirs, files in os.walk(indir):
        if "hpscan.pkl" in files:
            os.chdir(root)
            runnumber=root.split("/")[-1]
            copyCMD="cp hpscan.pkl {0}/{1}_hpscan.pkl".format(COLLECTIONPATH,runnumber)
            os.system(copyCMD)
    
if __name__=="__main__":
    
    PTYPES=["gamma"]#,"eplus","piplus"]

    for PTYPE in PTYPES:
        RAWINPATH="/Users/drdre/outputz/210923_parameterScan/{0}".format(PTYPE)
        VISPATH=os.path.join("/Users/drdre/outputz/210923_parameterScan/{0}/".format(PTYPE),"visualisations")
        
        cleanDirectories(indir=RAWINPATH)
        collectHistograms(indir=RAWINPATH)
        collectPkl(indir=RAWINPATH)