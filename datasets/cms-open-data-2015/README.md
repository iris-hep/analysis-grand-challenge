# Datasets from 2015 CMS Open Data release

Given a record id (e.g. `19980`), the list of all datasets can be found with the [`cernopendata-client`](https://cernopendata-client.readthedocs.io/):
```bash
cernopendata-client get-file-locations --protocol xrootd --recid 19980
```
Metadata can be extracted via `get-metadata`.

`datasets.txt` contains a list of folders to help identify relevant physics processes (`list-directory` with `cernopendata-client`).
Relevant records can be found on the [Open Data portal](https://opendata.cern.ch/).


## Samples categorized by process

- **ttbar**:
  - nominal:
    - [19980](https://opendata.cern.ch/record/19980): Powheg + Pythia 8 (ext3), 2413 files, 3.4 TB -> converted
    - [19981](https://opendata.cern.ch/record/19981): Powheg + Pythia 8 (ext4), 4653 files, 6.4 TB -> converted
  - scale variation:
    - [19982](https://opendata.cern.ch/record/19982): same as below, unclear if overlap
    - [19983](https://opendata.cern.ch/record/19983): Powheg + Pythia 8 "scaledown" (ext3), 902 files, 1.4 TB -> submitted
    - [19984](https://opendata.cern.ch/record/19984): same as below, unclear if overlap
    - [19985](https://opendata.cern.ch/record/19985): Powheg + Pythia 8 "scaleup" (ext3), 917 files, 1.3 TB -> submitted
  - ME variation:
    - [19977](https://opendata.cern.ch/record/19977): same as below, unclear if overlap
    - [19978](https://opendata.cern.ch/record/19978): aMC@NLO + Pythia 8 (ext1), 438 files, 647 GB -> submitted
  - PS variation:
    - [19999](https://opendata.cern.ch/record/19999): Powheg + Herwig++, 443 files, 810 GB -> submitted

- **single top**:
  - s-channel:
    - [19394](https://opendata.cern.ch/record/19394): aMC@NLO + Pythia 8, 114 files, 76 GB
  - t-channel:
    - [19406](https://opendata.cern.ch/record/19406): Powheg + Pythia 8 (antitop), 935 files, 1.1 TB
    - [19408](https://opendata.cern.ch/record/19408): Powheg + Pythia 8 (top), 1571 files, 1.8 TB
