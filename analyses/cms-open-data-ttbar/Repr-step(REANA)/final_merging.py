import uproot
import json

def extract_samples_from_json(json_file):
    output_files = []
    with open(json_file, 'r') as fd:
        data = json.load(fd)
        for sample, conditions in data.items():
            for condition in conditions:
                sample_name = f"everything_merged_{sample}__{condition}.root"
                output_files.append(sample_name)
    return output_files
json_file = "nanoaod_inputs.json"
# LIST_OF_FILES_PER_SAMPLE is a list of 9 files containing histograms per sample
LIST_OF_FILES_PER_SAMPLE = extract_samples_from_json(json_file)
print(LIST_OF_FILES_PER_SAMPLE)
with uproot.recreate("histograms_merged.root") as f_out:
    for h_file in LIST_OF_FILES_PER_SAMPLE:
        with uproot.open(h_file) as f_per_sample:
            for key in f_per_sample.keys(cycle=False):
                value = f_per_sample[key]
                f_out[key] = value
