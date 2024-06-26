import glob
import hist
import uproot

all_histograms = {}
for fname in glob.glob(f"histograms/histograms_{sample_name}_*/**/*.root", recursive=True):
    print(f"opening file {fname}")
    with uproot.open(fname) as f:
        # loop over all histograms in file
        for key in f.keys(cycle=False):
            if key not in all_histograms.keys():
                # this kind of histogram has not been seen yet, create a new entry for it
                all_histograms.update({key: hist.Hist(f[key])})
            else:
                # this kind of histogram is already being tracked, so add it
                all_histograms[key] += hist.Hist(f[key])
# save this to a new file
with uproot.recreate(f"everything_merged_{sample_name}.root") as f:
    for key, value in all_histograms.items():
        f[key] = value


file = uproot.open(f"everything_merged_{sample_name}.root")
keys = file.keys()
print(f"Keys for the everything_merged_{sample_name}.root:", keys)
