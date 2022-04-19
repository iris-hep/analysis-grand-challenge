# Datasets from 2015 CMS Open Data release

Given a record id (e.g. `19980`), the list of all datasets can be found with the [`cernopendata-client`](https://cernopendata-client.readthedocs.io/):
```bash
cernopendata-client get-file-locations --protocol xrootd --recid 19980
```
Metadata can be extracted via `get-metadata`.

`datasets.txt` contains a list of folders to help identify relevant physics processes (`list-directory` with `cernopendata-client`).
Relevant records can be found on the [Open Data portal](https://opendata.cern.ch/).

`create_file_list.sh` is a helper script to more conveniently create file lists for a list of records (using a local `cernopendata-client` installation).

`branches.txt` contains the branches in the converted ntuples.

`build_ntuple_json.py` creates a JSON file containing the converted ntuples, sorted by process / systematic variation.

The integrated data luminosity before quality requirements is `3378/pb`:
```bash
docker run -it --rm gitlab-registry.cern.ch/cms-cloud/brilws-docker\
 brilcalc lumi -c web --begin 256630 --end 260627
 ```
See the [luminosity doc page](http://opendata.cern.ch/docs/cms-guide-luminosity-calculation), and [information for applying quality requirements](https://opendata.cern.ch/record/14210).

## Samples categorized by process

- **ttbar**:
  - nominal:
    - [19980](https://opendata.cern.ch/record/19980): Powheg + Pythia 8 (ext3), 2413 files, 3.4 TB -> converted
    - [19981](https://opendata.cern.ch/record/19981): Powheg + Pythia 8 (ext4), 4653 files, 6.4 TB -> converted
  - scale variation:
    - [19982](https://opendata.cern.ch/record/19982): same as below, unclear if overlap
    - [19983](https://opendata.cern.ch/record/19983): Powheg + Pythia 8 "scaledown" (ext3), 902 files, 1.4 TB -> converted
    - [19984](https://opendata.cern.ch/record/19984): same as below, unclear if overlap
    - [19985](https://opendata.cern.ch/record/19985): Powheg + Pythia 8 "scaleup" (ext3), 917 files, 1.3 TB -> converted
  - ME variation:
    - [19977](https://opendata.cern.ch/record/19977): same as below, unclear if overlap
    - [19978](https://opendata.cern.ch/record/19978): aMC@NLO + Pythia 8 (ext1), 438 files, 647 GB -> converted
  - PS variation:
    - [19999](https://opendata.cern.ch/record/19999): Powheg + Herwig++, 443 files, 810 GB -> converted

- **single top**:
  - s-channel:
    - [19394](https://opendata.cern.ch/record/19394): aMC@NLO + Pythia 8, 114 files, 76 GB -> converted
  - t-channel:
    - [19406](https://opendata.cern.ch/record/19406): Powheg + Pythia 8 (antitop), 935 files, 1.1 TB -> converted
    - [19408](https://opendata.cern.ch/record/19408): Powheg + Pythia 8 (top), 1571 files, 1.8 TB -> converted
  - tW:
    - nominal:
      - [19412](https://opendata.cern.ch/record/19412): Powheg + Pythia 8 (antitop), 27 files, 30 GB -> converted
      - [19419](https://opendata.cern.ch/record/19419): Powheg + Pythia 8 (top), 23 files, 30 GB -> converted
    - DS:
      - [19410](https://opendata.cern.ch/record/19410): Powheg + Pythia 8 DS (antitop), 13 files, 15 GB
      - [19417](https://opendata.cern.ch/record/19417): Powheg + Pythia 8 DS (top), 13 files, 14 GB
    - scale variations:
      - [19415](https://opendata.cern.ch/record/19415): Powheg + Pythia 8 "scaledown" (antitop), 11 files, 15 GB
      - [19422](https://opendata.cern.ch/record/19422): Powheg + Pythia 8 "scaledown" (top), 13 files, 15 GB
      - [19416](https://opendata.cern.ch/record/19416): Powheg + Pythia 8 "scaleup" (antitop), 12 files, 14 GB
      - [19423](https://opendata.cern.ch/record/19423): Powheg + Pythia 8 "scaleup" (top), 13 files, 14 GB

    - there are also larger `NoFullyHadronicDecays` samples: [19411](https://opendata.cern.ch/record/19411), [19418](https://opendata.cern.ch/record/19418)
  - tZ / tWZ: potentially missing in inputs, not included in `/ST_*`

- **W+jets**:
  - nominal (with 1l filter):
    - [20546](https://opendata.cern.ch/record/20546): same as below, unclear if overlap
    - [20547](https://opendata.cern.ch/record/20547): aMC@NLO + Pythia 8 (ext2), 5601 files, 4.5 TB -> converted
    - [20548](https://opendata.cern.ch/record/20548): aMC@NLO + Pythia 8 (ext4), 4598 files, 3.8 TB -> converted

- **data**:
  - single muon:
    - [24119](https://opendata.cern.ch/record/24119): 1916 files, 1.4 TB -> converted
  - single electron:
    - [24120](https://opendata.cern.ch/record/24120): 2974 files, 2.6 TB -> converted
  - validated runs:
    - [14210](https://opendata.cern.ch/record/14210): single txt file
