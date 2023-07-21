.. _versions-description:

AGC Analysis Task Versions
================================

The below table gives a brief overview of the AGC versions. Each version here corresponds to a slightly altered task.

.. list-table:: Versions
   :widths: 12 22 22 22 22
   :header-rows: 1
   
   * - Version
     - Datasets
     - Cuts
     - Systematics
     - Machine Learning
   * - 0
     - CMS 2015 Open Data (POET)
     - Exactly one lepton with :math:`p_T>25` GeV; at least four jets with :math:`p_T>25` GeV; at least one jet with :math:`b`-tag > 0.5
     - :math:`t\bar{t}` sample variations, ``pt_scale`` variations, ``pt_res`` variations, ``btag`` variations, **W + jets** scale variations, luminosity
     - None
   * - 1
     - CMS 2015 Open Data (NanoAOD)
     - Exactly one lepton with :math:`p_T>25` GeV; at least four jets with :math:`p_T>25` GeV; at least one jet with :math:`b`-tag > 0.5
     - :math:`t\bar{t}` sample variations, ``pt_scale`` variations, ``pt_res`` variations, ``btag`` variations, **W + jets** scale variations, luminosity
     - None
   * - 2 (WIP)
     - CMS 2015 Open Data (NanoAOD)
     - Exactly one lepton with :math:`p_T>30` GeV; at least four jets with :math:`p_T>30` GeV; at least one jet with :math:`b`-tag > 0.5 (see :ref:`versions-cuts` for additional cuts)
     - :math:`t\bar{t}` sample variations, ``pt_scale`` variations, ``pt_res`` variations, ``btag`` variations, **W + jets** scale variations, luminosity
     - BDT to predict jet-parton assignment in :math:`t\bar{t}` events
     
     
Reference Implementation Versions 
================================
This section is specific to the implementation in the `main repository <https://github.com/iris-hep/analysis-grand-challenge>`_.

The below table gives a brief overview of the different tags of the reference implementation, including descriptions of minor versions and patches which are implementation-specific. 
Note that the major versions (0, 1, and 2) correspond to differences in analysis task (described above), while minor versions are reserved for individual implementations to assign for small changes and patches. 
Our reference implementation for each major task (0, 1, 2) will always be the latest tag within that series.

.. list-table:: Tags
   :widths: 8 5 29 29 29
   :header-rows: 1
   
   * - Tag
     - Version
     - Available Pipelines
     - Systematics
     - Dependency Management
   * - 0.1.0
     - 0
     - Pure ``coffea``; ``coffea`` with ``ServiceX`` processor
     - Systematic variations within ``coffea`` processor are manually calculated using ``awkward`` array logic (jet :math:`p_T` variations are not propagated through signal region observable calculation)
     - Functions used in ``coffea`` processor are defined in the notebook
   * - 0.2.0
     - 0
     - Pure ``coffea``; create cached files using ``ServiceX`` queries followed by standalone ``coffea`` processing
     - Systematic variations within ``coffea`` processor are manually calculated using ``awkward`` array logic (jet :math:`p_T` variations are not propagated through signal region observable calculation)
     - Functions used in ``coffea`` processor are defined in the notebook
   * - 1.0.0
     - 1
     - Pure ``coffea``; create cached files using ``ServiceX`` queries followed by standalone ``coffea`` processing
     - Systematic variations within ``coffea`` processor are manually calculated using ``awkward`` array logic (jet :math:`p_T` variations are not propagated through signal region observable calculation)
     - Functions used in ``coffea`` processor are defined in the notebook
   * - 1.1.0
     - 1
     - Pure ``coffea``; create cached files using ``ServiceX`` queries followed by standalone ``coffea`` processing
     - Systematic variations within ``coffea`` processor are manually calculated using ``awkward`` array logic (jet :math:`p_T` variations corrected)
     - Functions used in ``coffea`` processor are defined in the notebook
   * - 1.2.0
     - 1
     - Pure ``coffea``; create cached files using ``ServiceX`` queries followed by standalone ``coffea`` processing
     - Systematic variations within ``coffea`` processor are manually calculated using ``awkward`` array logic (b-tagging cuts corrected, no 1e-6 offsetting of histogram yields)
     - Functions used in ``coffea`` processor are defined in the notebook
   * - 2.0.0 (WIP)
     - 2
     - Pure ``coffea``; create cached files using ``ServiceX`` queries followed by standalone ``coffea`` processing; optional machine learning component (with additional option to use ``NVIDIA Triton`` inference server)
     - Systematic variations within ``coffea`` processor handled by ``correctionlib``
     - Modules are shipped to ``dask`` workers using ``cloudpickle``


Datasets
================================

The datasets used for the CMS :math:`t\bar{t}` notebook are from the 2015 CMS Open Data release. 
Versions 0.1.0 and 0.2.0 use ntuples generated using the `Physics Objects Extractor Tool (POET) <https://github.com/cms-opendata-analyses/PhysObjectExtractorTool>`_.

All versions >=1.0.0 use NanoAOD instead. 
The NanoAOD was generated from the 2015 CMS Open Data release using this pull request of CMSSW: `https://github.com/cms-sw/cmssw/pull/39040 <https://github.com/cms-sw/cmssw/pull/39040>`_. 
To set this up, the following commands should be run::
    
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    scram list CMSSW_10_6_
    scram project CMSSW_10_6_30
    cd CMSSW_10_6_30/
    cmsenv
    cd src/
    git cms-merge-topic 39040
    ls -al
    scram build -j5

From this point, for data, you can use::

    cmsDriver.py --python_filename doublemuon_cfg.py --eventcontent NANOAOD --customise Configuration/DataProcessing/Utils.addMonitoring --datatier NANOAOD --fileout file:doublemuon_nanoaod.root --conditions 106X_dataRun2_v36 --step NANO --filein file:doublemuon_miniaod.root --era Run2_25ns,run2_nanoAOD_106X2015 --no_exec --data -n -1
    
For MC, you can use::
    
    cmsDriver.py --python_filename nanoaod15_cfg.py --eventcontent NANOAODSIM --customise Configuration/DataProcessing/Utils.addMonitoring --datatier NANOAODSIM --fileout file:nanoaod15.root --conditions 102X_mcRun2_asymptotic_v8 --step NANO --filein file:miniaod2015.root --era Run2_25ns,run2_nanoAOD_106X2015 --no_exec --mc -n -1

The code used to generate and subsequently merge these files is located in the following repository: `https://github.com/ekauffma/produce-nanoAODs <https://github.com/ekauffma/produce-nanoAODs>`_

The data used is the same, regardless of MiniAOD vs NanoAOD. 
The list of datasets separated by process is included below:

* **ttbar**:

  * nominal:
    
    * `19980 <https://opendata.cern.ch/record/19980>`_: Powheg + Pythia 8 (ext3), 2413 files, 3.4 TB -> converted
    * `19981 <https://opendata.cern.ch/record/19981>`_: Powheg + Pythia 8 (ext4), 4653 files, 6.4 TB -> converted
    
  * scale variation:
    
    * `19982 <https://opendata.cern.ch/record/19982>`_: same as below, unclear if overlap
    * `19983 <https://opendata.cern.ch/record/19983>`_: Powheg + Pythia 8 "scaledown" (ext3), 902 files, 1.4 TB -> converted
    * `19984 <https://opendata.cern.ch/record/19984>`_: same as below, unclear if overlap
    * `19985 <https://opendata.cern.ch/record/19985>`_: Powheg + Pythia 8 "scaleup" (ext3), 917 files, 1.3 TB -> converted
  
  * ME variation:
    
    * `19977 <https://opendata.cern.ch/record/19977>`_: same as below, unclear if overlap
    * `19978 <https://opendata.cern.ch/record/19978>`_: aMC@NLO + Pythia 8 (ext1), 438 files, 647 GB -> converted
  
  * PS variation:
    
    * `19999 <https://opendata.cern.ch/record/19999>`_: Powheg + Herwig++, 443 files, 810 GB -> converted

* **single top**:

  * s-channel:
    
    * `19394 <https://opendata.cern.ch/record/19394>`_: aMC@NLO + Pythia 8, 114 files, 76 GB -> converted
  
  * t-channel:
    
    * `19406 <https://opendata.cern.ch/record/19406>`_: Powheg + Pythia 8 (antitop), 935 files, 1.1 TB -> converted
    * `19408 <https://opendata.cern.ch/record/19408>`_: Powheg + Pythia 8 (top), 1571 files, 1.8 TB -> converted
  
  * tW:
    
    * nominal:
      
      * `19412 <https://opendata.cern.ch/record/19412>`_: Powheg + Pythia 8 (antitop), 27 files, 30 GB -> converted
      * `19419 <https://opendata.cern.ch/record/19419>`_: Powheg + Pythia 8 (top), 23 files, 30 GB -> converted
    
    * DS:
      
      * `19410 <https://opendata.cern.ch/record/19410>`_: Powheg + Pythia 8 DS (antitop), 13 files, 15 GB
      * `19417 <https://opendata.cern.ch/record/19417>`_: Powheg + Pythia 8 DS (top), 13 files, 14 GB
    
    * scale variations:
      
      * `19415 <https://opendata.cern.ch/record/19415>`_: Powheg + Pythia 8 "scaledown" (antitop), 11 files, 15 GB
      * `19422 <https://opendata.cern.ch/record/19422>`_: Powheg + Pythia 8 "scaledown" (top), 13 files, 15 GB
      * `19416 <https://opendata.cern.ch/record/19416>`_: Powheg + Pythia 8 "scaleup" (antitop), 12 files, 14 GB
      * `19423 <https://opendata.cern.ch/record/19423>`_: Powheg + Pythia 8 "scaleup" (top), 13 files, 14 GB

    * there are also larger `NoFullyHadronicDecays` samples: `19411 <https://opendata.cern.ch/record/19411>`_, `19418 <https://opendata.cern.ch/record/19418>`_
  
  * tZ / tWZ: potentially missing in inputs, not included in `/ST_*`

* **W+jets**:

  * nominal (with 1l filter):
  
    * `20546 <https://opendata.cern.ch/record/20546>`_: same as below, unclear if overlap
    * `20547 <https://opendata.cern.ch/record/20547>`_: aMC@NLO + Pythia 8 (ext2), 5601 files, 4.5 TB -> converted
    * `20548 <https://opendata.cern.ch/record/20548>`_: aMC@NLO + Pythia 8 (ext4), 4598 files, 3.8 TB -> converted

* **data**:

  * single muon:
  
    * `24119 <https://opendata.cern.ch/record/24119>`_: 1916 files, 1.4 TB -> converted
  
  * single electron:
    
    * `24120 <https://opendata.cern.ch/record/24120>`_: 2974 files, 2.6 TB -> converted
  
  * validated runs:
    
    * `24210 <https://opendata.cern.ch/record/24210>`_: single txt file
    
More information about datasets can be found in `analysis-grand-challenge/datasets/cms-open-data-2015/ <https://github.com/iris-hep/analysis-grand-challenge/tree/main/datasets/cms-open-data-2015>`_.


.. _versions-cuts:

Cuts
================================

For versions 0.1.0, 0.2.0, and 1.0.0, the cuts used are the following:

* Leptons (electrons and muons) must have :math:`p_T>25` GeV
* Events must contain exactly one lepton
* Jets must have :math:`p_T>25` GeV
* Events must have at least four jets
* Jets are considered :math:`b`-tagged if they have a :math:`b`-tag score over `B_TAG_THRESHOLD=0.5`.
* Events must have at least one :math:`b`-tagged jet
* **4j1b Region**: Events must have exactly one :math:`b`-tagged jet
* **4j2b Region**: Events must have two or more :math:`b`-tagged jets

This is modified to better reflect common practices in CMS in subsequent versions, using the following cuts:

* Leptons (electrons and muons) must have :math:`p_T>30` GeV, :math:`|\eta|<2.1`, and ``sip3d<4`` (significance of 3d impact parameter)
* For electrons, we also require ``cutBased==4`` (tight)
* For muons, we also require ``tightId`` and ``pfRelIso04_all<0.15`` (PF relative isolation dR=0.4, total (deltaBeta corrections))
* Events must contain exactly one lepton
* Jets must have :math:`p_T>30` GeV and :math:`|\eta|<2.4` and satisfy a tight lepton veto (``isTightLeptonVeto``, or ``jetId==6``)
* Events must have at least four jets
* Jets are considered :math:`b`-tagged if they have a :math:`b`-tag score over `B_TAG_THRESHOLD=0.5`.
* Events must have at least one :math:`b`-tagged jet
* **4j1b Region**: Events must have exactly one :math:`b`-tagged jet
* **4j2b Region**: Events must have two or more :math:`b`-tagged jets
