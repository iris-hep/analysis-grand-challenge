# Compare the content of histograms produced by ttbar_analysis_pipeline with a reference file.
# A reference file for N_FILES_MAX_PER_SAMPLE=1 is available in directory `reference/`.

from __future__ import annotations
import argparse
from collections import defaultdict
import json
import numpy as np
import sys
import uproot

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--histos", help="ROOT file containing the output histograms. Defaults to './histograms.root'.", default="histograms.root")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--reference", help="JSON reference against which histogram contents should be compared")
    group.add_argument("--dump-json", help="Print JSON representation of histogram contents to screen", action='store_true')
    return parser.parse_args()

# convert uproot file containing only TH1Ds to a corresponding JSON-compatible dict with structure:
# { "histo1": { "edges": [...], "contents": [...] }, "histo2": { ... }, ... }
# Only the highest namecycle for every histogram is considered, and cycles are stripped from the histogram names.
def as_dict(f: uproot.ReadOnlyDirectory) -> dict[str, dict]:
    histos = defaultdict(dict)
    cycles: dict(str, int) = {}
    for k, v in f.items():
        assert isinstance(v, uproot.behaviors.TH1.Histogram)
        # this assumes that the rightmost ";" (if any) comes before a namecycle
        name, *cycle = k.rsplit(";", 1)
        cycle = int(cycle[0]) if len(cycle) > 0 else -1
        if name in histos and cycles["cycle"] > cycle:
            continue # found a lower cycle for a histogram we already have

        cycles[name] = cycle
        histos[name]["edges"] = v.axis().edges().tolist()
        histos[name]["contents"] = v.counts(flow=True).tolist()
    return histos

def validate(histos: dict, reference: dict) -> dict[str, list[str]]:
    errors = defaultdict(list)
    for name, ref_h in reference.items():
        if name not in histos:
            errors[name].append("Histogram not found.")
            continue

        h = histos[name]
        if not np.allclose(h['edges'], ref_h['edges']):
            errors[name].append(f"Edges do not match:\n\tgot      {h['edges']}\n\texpected {ref_h['edges']}")
        contents_depend_on_rng = "pt_res_up" in name # skip checking the contents of these histograms as they are not stable
        if not contents_depend_on_rng and not np.allclose(h['contents'], ref_h['contents']):
            errors[name].append(f"Contents do not match:\n\tgot      {h['contents']}\n\texpected {ref_h['contents']}")

    return errors

if __name__ == "__main__":
    args = parse_args()
    with uproot.open(args.histos) as f:
        histos = as_dict(f)

    if args.dump_json:
        print(json.dumps(histos))
        sys.exit(0)

    with open(args.reference) as reference:
        ref_histos = json.load(reference)

    print(f"Validating '{args.histos}' against reference '{args.reference}'...")
    errs = validate(histos=histos, reference=ref_histos)
    if len(errs) == 0:
        print("All good!")
    else:
        for hist_name, errors in errs.items():
            errors = '\n\t'.join(errors)
            print(f"{hist_name}\n\t{errors}")
        sys.exit(1)
