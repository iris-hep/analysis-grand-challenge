config = {
    "global": {
        # ServiceX: ignore cache with repeated queries
        "SERVICEX_IGNORE_CACHE": False,
        # analysis facility: set to "coffea_casa" for coffea-casa environments, "EAF" for FNAL, "local" for local setups
        "AF": "coffea_casa",
        # number of bins for standard histograms in processor
        "NUM_BINS": 25,
        # lower end of standard histograms in processor
        "BIN_LOW": 50,
        # upper end of standard histograms in processor
        "BIN_HIGH": 550,

    },
    "benchmarking": {
        # chunk size to use
        "CHUNKSIZE": 200000,
        # read files from public EOS (thanks to the CMS DPOA team!)
        # note that they are likely only available temporarily
        # and not part of an official CMS Open Data release
        "INPUT_FROM_EOS": True,
        # prefix for URIs for ATLAS-style xcache use
        # e.g. "root://xcache.af.uchicago.edu//" for UChicago
        "XCACHE_ATLAS_PREFIX": None,
        ### metadata to propagate through to metrics ###
        # "ssl-dev" allows for the switch to local data on /data
        "AF_NAME": "coffea_casa",
        # currently has no effect
        "SYSTEMATICS": "all",
        # does not do anything, only used for metric gathering (set to 2 for distributed coffea-casa)
        "CORES_PER_WORKER": 2,
        # scaling for local setups with FuturesExecutor
        "NUM_CORES": 4,
        # only I/O, all other processing disabled
        "DISABLE_PROCESSING": False,
        ### read additional branches (only with DISABLE_PROCESSING = True) ###
        # acceptable values are 4.1, 15, 25, 50 (corresponding to % of file read), 4.1% corresponds to the standard branches used in the notebook
        "IO_FILE_PERCENT": "4.1",
        # nanoAOD branches that correspond to different values of IO_FILE_PERCENT
        "IO_BRANCHES": {
            "4.1": [
                "Jet_pt",
                "Jet_eta",
                "Jet_phi",
                "Jet_btagCSVV2",
                "Jet_mass",
                "Muon_pt",
                "Electron_pt",
            ],
            "15": ["LHEPdfWeight", "GenPart_pdgId", "CorrT1METJet_phi"],
            "25": [
                "LHEPdfWeight",
                "GenPart_pt",
                "GenPart_eta",
                "GenPart_pdgId",
                "LHEScaleWeight",
            ],
            "50": [
                "LHEPdfWeight",
                "GenPart_pt",
                "GenPart_eta",
                "GenPart_phi",
                "GenPart_pdgId",
                "GenPart_genPartIdxMother",
                "GenPart_statusFlags",
                "GenPart_mass",
                "LHEScaleWeight",
                "GenJet_pt",
                "GenPart_status",
                "LHEPart_eta",
                "LHEPart_phi",
                "LHEPart_pt",
                "GenJet_eta",
                "GenJet_phi",
                "Jet_eta",
                "Jet_phi",
                "SoftActivityJet_pt",
                "SoftActivityJet_phi",
                "SoftActivityJet_eta",
                "GenJet_mass",
                "Jet_pt",
                "Jet_mass",
                "LHEPart_mass",
                "Jet_qgl",
                "Jet_muonSubtrFactor",
                "Jet_puIdDisc",
            ],
        },
    },
    "ml": {
        # maximum number of jets to consider in jet parton assignment task
        "MAX_N_JETS": 6,
        # name of ML model in Triton server
        "MODEL_NAME": "reconstruction_bdt_xgb",
        # URL of Triton server
        "TRITON_URL": "triton-inference-server:8001",
        # Triton model version which is trained on even events
        "MODEL_VERSION_EVEN": "2",
        # Triton model version which is trained on odd events
        "MODEL_VERSION_ODD": "1",
        # path to local model (trained on even events) if not using Triton
        "XGBOOST_MODEL_PATH_EVEN": "models/model_even.json",
        # path to local model (trained on odd events) if not using Triton
        "XGBOOST_MODEL_PATH_ODD": "models/model_odd.json",
        # histogram bin lower limit to use for each ML input feature
        "BIN_LOW": [
            0,
            0,
            0,
            0,
            50,
            50,
            50,
            50,
            25,
            25,
            25,
            25,
            0,
            0,
            0,
            0,
            -1,
            -1,
            -1,
            -1,
        ],
        # histogram bin upper limit to use for each ML input feature
        "BIN_HIGH": [
            6,
            6,
            6,
            6,
            300,
            300,
            550,
            550,
            300,
            300,
            300,
            300,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        # names of each ML input feature (used when creating histograms)
        "FEATURE_NAMES": [
            "deltar_leptonbtoplep",
            "deltar_w1w2",
            "deltar_w1btophad",
            "deltar_w2btophad",
            "mass_leptonbtoplep",
            "mass_w1w2",
            "mass_w1w2btophad",
            "pt_w1w2btophad",
            "pt_w1",
            "pt_w2",
            "pt_btophad",
            "pt_btoplep",
            "btag_w1",
            "btag_w2",
            "btag_btophad",
            "btag_btoplep",
            "qgl_w1",
            "qgl_w2",
            "qgl_btophad",
            "qgl_btoplep",
        ],
        # labels for each ML input feature (used for plotting)
        "FEATURE_DESCRIPTIONS": [
            "Delta R between $b_{top-lep}$ Jet and Lepton",
            "Delta R between the two W Jets",
            "Delta R between first W Jet and $b_{top-had}$ Jet",
            "Delta R between second W Jet and $b_{top-had}$ Jet",
            "Combined Mass of $b_{top-lep}$ Jet and Lepton [GeV]",
            "Combined Mass of the two W Jets [GeV]",
            "Combined Mass of $b_{top-had}$ Jet and the two W Jets [GeV]",
            "Combined $p_T$ of $b_{top-had}$ Jet and the two W Jets [GeV]",
            "$p_T$ of the first W Jet [GeV]",
            "$p_T$ of the second W Jet [GeV]",
            "$p_T$ of the $b_{top-had}$ Jet [GeV]",
            "$p_T$ of the $b_{top-lep}$ Jet [GeV]",
            "btagCSVV2 of the first W Jet",
            "btagCSVV2 of the second W Jet",
            "btagCSVV2 of the $b_{top-had}$ Jet",
            "btagCSVV2 of the $b_{top-lep}$ Jet",
            "Quark vs Gluon likelihood discriminator of the first W Jet",
            "Quark vs Gluon likelihood discriminator of the second W Jet",
            "Quark vs Gluon likelihood discriminator of the $b_{top-had}$ Jet",
            "Quark vs Gluon likelihood discriminator of the $b_{top-lep}$ Jet",
        ],
    },
}
