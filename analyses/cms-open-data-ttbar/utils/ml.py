import awkward as ak
import numpy as np
from xgboost import XGBClassifier
from .config import config
import vector


# local loading of ML models
model_even = None
model_odd = None


def load_models():
    global model_even
    model_even = XGBClassifier()
    model_even.load_model(config["ml"]["XGBOOST_MODEL_PATH_EVEN"])

    global model_odd
    model_odd = XGBClassifier()
    model_odd.load_model(config["ml"]["XGBOOST_MODEL_PATH_ODD"])


def get_permutations_dict(MAX_N_JETS, include_labels=False, include_eval_mat=False):
    """
    Get dictionary which lists the different permutations considered for each event, depending on number of jets in an event.

    Args:
        MAX_N_JETS: maximum number of jets to consider for permutations (ordered by pT)
        include_labels: whether to include another dictionary with associated labels for each permutation
                        (24=W jet, 6=top_hadron jet, -6=top_lepton jet)
        include_eval_mat: whether to include a dictionary of matrices which calculates the associated jet score between predicted and truth labels

    Returns:
        permutations_dict: dictionary containing lists of permutation indices for each number of jets up to MAX_N_JETS
        labels_dict (optional): dictionary containing associated labels of permutations listed in permutations_dict
        evaluation_matrices (optional): dictionary containing associated evaluation matrices to get the associated jet score using the predicted and truth labels
    """

    # calculate the dictionary of permutations for each number of jets
    permutations_dict = {}
    for n in range(4, MAX_N_JETS + 1):
        test = ak.Array(range(n))
        unzipped = ak.unzip(ak.argcartesian([test] * 4, axis=0))

        combos = ak.combinations(ak.Array(range(4)), 2, axis=0)
        different = unzipped[combos[0]["0"]] != unzipped[combos[0]["1"]]
        for i in range(1, len(combos)):
            different = different & (unzipped[combos[i]["0"]] != unzipped[combos[i]["1"]])
        permutations = ak.zip([test[unzipped[i][different]] for i in range(len(unzipped))], depth_limit=1).tolist()

        permutations = ak.concatenate([test[unzipped[i][different]][..., np.newaxis] for i in range(len(unzipped))], axis=1).to_list()

        permutations_dict[n] = permutations

    # for each permutation, calculate the corresponding label
    labels_dict = {}
    for n in range(4, MAX_N_JETS + 1):
        current_labels = []
        for inds in permutations_dict[n]:
            inds = np.array(inds)
            current_label = 100 * np.ones(n)
            current_label[inds[:2]] = 24
            current_label[inds[2]] = 6
            current_label[inds[3]] = -6
            current_labels.append(current_label.tolist())

        labels_dict[n] = current_labels

    # get rid of duplicates since we consider W jets to be exchangeable
    # (halves the number of permutations we consider)
    for n in range(4, MAX_N_JETS + 1):
        res = []
        for idx, val in enumerate(labels_dict[n]):
            if val in labels_dict[n][:idx]:
                res.append(idx)
        labels_dict[n] = np.array(labels_dict[n])[res].tolist()
        permutations_dict[n] = np.array(permutations_dict[n])[res].tolist()

    if include_labels and not include_eval_mat:
        return permutations_dict, labels_dict

    elif include_labels and include_eval_mat:
        # these matrices tell you the overlap between the predicted label (rows) and truth label (columns)
        # the "score" in each matrix entry is the number of jets which are assigned correctly
        evaluation_matrices = {}  # overall event score

        for n in range(4, MAX_N_JETS + 1):
            evaluation_matrix = np.zeros((len(permutations_dict[n]), len(permutations_dict[n])))

            for i in range(len(permutations_dict[n])):
                for j in range(len(permutations_dict[n])):
                    evaluation_matrix[i, j] = sum(np.equal(labels_dict[n][i], labels_dict[n][j]))

            evaluation_matrices[n] = evaluation_matrix / 4

        return permutations_dict, labels_dict, evaluation_matrices

    else:
        return permutations_dict


def get_features(jets, electrons, muons, max_n_jets=6):
    """
    Calculate features for each of the 12 combinations per event

    Args:
        jets: selected jets
        electrons: selected electrons
        muons: selected muons
        permutations_dict: which permutations to consider for each number of jets in an event

    Returns:
        features (flattened to remove event level)
        perm_counts: how many permutations in each event. use to unflatten features
    """
    # for four-vector addition of custom Momentum4D instances below
    vector.register_awkward()

    permutations_dict = get_permutations_dict(max_n_jets)

    # calculate number of jets in each event
    njet = ak.num(jets).to_numpy()
    # don't consider every jet for events with high jet multiplicity
    njet[njet > max(permutations_dict.keys())] = max(permutations_dict.keys())
    # create awkward array of permutation indices
    perms = ak.Array([permutations_dict[n] for n in njet])
    perm_counts = ak.num(perms)

    #### calculate features ####
    features = np.zeros((sum(perm_counts), 20))

    # grab lepton info
    leptons = ak.flatten(ak.concatenate((electrons, muons), axis=1), axis=-1)

    # delta R between b_toplep and lepton
    features[:, 0] = ak.flatten(np.sqrt((leptons.eta - jets[perms[..., 3]].eta) ** 2
                                        + (leptons.phi - jets[perms[..., 3]].phi) ** 2)).to_numpy()

    # delta R between the two W
    features[:, 1] = ak.flatten(np.sqrt((jets[perms[..., 0]].eta - jets[perms[..., 1]].eta) ** 2
                                        + (jets[perms[..., 0]].phi - jets[perms[..., 1]].phi) ** 2)).to_numpy()

    # delta R between W and b_tophad
    features[:, 2] = ak.flatten(np.sqrt((jets[perms[..., 0]].eta - jets[perms[..., 2]].eta) ** 2
                                        + (jets[perms[..., 0]].phi - jets[perms[..., 2]].phi) ** 2)).to_numpy()
    features[:, 3] = ak.flatten(np.sqrt((jets[perms[..., 1]].eta - jets[perms[..., 2]].eta) ** 2
                                        + (jets[perms[..., 1]].phi - jets[perms[..., 2]].phi) ** 2)).to_numpy()

    # combining the original leptons and jets arrays results in column overtouching
    # see https://github.com/CoffeaTeam/coffea/issues/892 for details
    # to work around this, create smaller versions of the arrays using only the relevant
    # four-vector information
    el_p4 = ak.zip({"pt": electrons.pt, "eta": electrons.eta, "phi": electrons.phi, "mass": electrons.mass}, with_name="Momentum4D")
    mu_p4 = ak.zip({"pt": muons.pt, "eta": muons.eta, "phi": muons.phi, "mass": muons.mass}, with_name="Momentum4D")
    lep_p4 = ak.flatten(ak.concatenate((el_p4, mu_p4), axis=1), axis=-1)
    jet_p4 = ak.zip({"pt": jets.pt, "eta": jets.eta, "phi": jets.phi, "mass": jets.mass}, with_name="Momentum4D")

    # combined mass of b_toplep and lepton
    features[:, 4] = ak.flatten((lep_p4 + jet_p4[perms[..., 3]]).mass).to_numpy()

    # combined mass of W
    features[:, 5] = ak.flatten((jets[perms[..., 0]] + jets[perms[..., 1]]).mass).to_numpy()

    # combined mass of W and b_tophad
    features[:, 6] = ak.flatten((jets[perms[..., 0]] + jets[perms[..., 1]] + jets[perms[..., 2]]).mass).to_numpy()

    # combined pT of W and b_tophad
    features[:, 7] = ak.flatten((jets[perms[..., 0]] + jets[perms[..., 1]] + jets[perms[..., 2]]).pt).to_numpy()

    # pt of every jet
    features[:, 8] = ak.flatten(jets[perms[..., 0]].pt).to_numpy()
    features[:, 9] = ak.flatten(jets[perms[..., 1]].pt).to_numpy()
    features[:, 10] = ak.flatten(jets[perms[..., 2]].pt).to_numpy()
    features[:, 11] = ak.flatten(jets[perms[..., 3]].pt).to_numpy()

    # btagCSVV2 of every jet
    features[:, 12] = ak.flatten(jets[perms[..., 0]].btagCSVV2).to_numpy()
    features[:, 13] = ak.flatten(jets[perms[..., 1]].btagCSVV2).to_numpy()
    features[:, 14] = ak.flatten(jets[perms[..., 2]].btagCSVV2).to_numpy()
    features[:, 15] = ak.flatten(jets[perms[..., 3]].btagCSVV2).to_numpy()

    # quark-gluon likelihood discriminator of every jet
    features[:, 16] = ak.flatten(jets[perms[..., 0]].qgl).to_numpy()
    features[:, 17] = ak.flatten(jets[perms[..., 1]].qgl).to_numpy()
    features[:, 18] = ak.flatten(jets[perms[..., 2]].qgl).to_numpy()
    features[:, 19] = ak.flatten(jets[perms[..., 3]].qgl).to_numpy()

    return features, perm_counts


def get_inference_results_local(features, even, model_even, model_odd):
    results = np.zeros(features.shape[0])
    if len(features[even]) > 0:
        results[even] = model_odd.predict_proba(features[even, :])[:, 1]
    if len(features[np.invert(even)]) > 0:
        results[np.invert(even)] = model_even.predict_proba(
            features[np.invert(even), :]
        )[:, 1]
    return results


def get_inference_results_triton(features, even, triton_client, MODEL_NAME,
                                 MODEL_VERS_EVEN, MODEL_VERS_ODD):

    model_metadata = triton_client.get_model_metadata(MODEL_NAME, MODEL_VERS_EVEN)

    input_name = model_metadata.inputs[0].name
    dtype = model_metadata.inputs[0].datatype
    output_name = model_metadata.outputs[0].name

    results = np.zeros(features.shape[0])

    import tritonclient.grpc as grpcclient

    output = grpcclient.InferRequestedOutput(output_name)

    if len(features[even]) > 0:
        inpt = [grpcclient.InferInput(input_name, features[even].shape, dtype)]
        inpt[0].set_data_from_numpy(features[even].astype(np.float32))
        results[even] = triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERS_EVEN,
            inputs=inpt,
            outputs=[output],
        ).as_numpy(output_name)[:, 1]
    if len(features[np.invert(even)]) > 0:
        inpt = [
            grpcclient.InferInput(input_name, features[np.invert(even)].shape, dtype)
        ]
        inpt[0].set_data_from_numpy(features[np.invert(even)].astype(np.float32))
        results[np.invert(even)] = triton_client.infer(
            model_name=MODEL_NAME,
            model_version=MODEL_VERS_ODD,
            inputs=inpt,
            outputs=[output],
        ).as_numpy(output_name)[:, 1]

    return results

def training_filter(jets, electrons, muons, genparts, even):
    '''
    Filters events down to training set and calculates jet-level labels
    
    Args:
        jets: selected jets after region filter (and selecting leading four for each event)
        electrons: selected electrons after region filter
        muons: selected muons after region filter
        genparts: selected genpart after region filter
        even: whether the event is even-numbered (used to separate training events)
    
    Returns:
        jets: selected jets after training filter
        electrons: selected electrons after training filter
        muons: selected muons after training filter
        labels: labels of jets within an event (24=W, 6=top_hadron, -6=top_lepton)
        even: whether the event is even-numbered
    '''
    #### filter genPart to valid matching candidates ####

    # get rid of particles without parents
    genpart_parent = genparts.distinctParent
    genpart_filter = np.invert(ak.is_none(genpart_parent, axis=1))
    genparts = genparts[genpart_filter]
    genpart_parent = genparts.distinctParent

    # ensure that parents are top quark or W
    genpart_filter2 = ((np.abs(genpart_parent.pdgId)==6) | (np.abs(genpart_parent.pdgId)==24))
    genparts = genparts[genpart_filter2]

    # ensure particle itself is a quark
    genpart_filter3 = ((np.abs(genparts.pdgId)<7) & (np.abs(genparts.pdgId)>0))
    genparts = genparts[genpart_filter3]

    # get rid of duplicates
    genpart_filter4 = genparts.hasFlags("isLastCopy")
    genparts = genparts[genpart_filter4]
            
        
    #### get jet-level labels and filter events to training set
        
    # match jets to nearest valid genPart candidate
    nearest_genpart = jets.nearest(genparts, threshold=0.4)
    nearest_parent = nearest_genpart.distinctParent # parent of matched particle
    parent_pdgid = nearest_parent.pdgId # pdgId of parent particle
    grandchild_pdgid = nearest_parent.distinctChildren.distinctChildren.pdgId # pdgId of particle's parent's grandchildren

    grandchildren_flat = np.abs(ak.flatten(grandchild_pdgid,axis=-1)) # flatten innermost axis for convenience

    # if particle has a cousin that is a lepton
    has_lepton_cousin = (ak.sum(((grandchildren_flat%2==0) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                axis=-1)>0)
    # if particle has a cousin that is a neutrino
    has_neutrino_cousin = (ak.sum(((grandchildren_flat%2==1) & (grandchildren_flat>10) & (grandchildren_flat<19)),
                                  axis=-1)>0)

    # if a particle has a lepton cousin and a neutrino cousin
    has_both_cousins = ak.fill_none((has_lepton_cousin & has_neutrino_cousin), False).to_numpy()

    # get labels from parent pdgId (fill none with 100 to filter out events with those jets)
    labels = np.abs(ak.fill_none(parent_pdgid,100).to_numpy())
    labels[has_both_cousins] = -6 # assign jets with both cousins as top_lepton (not necessarily antiparticle)

    training_event_filter = (np.sum(labels,axis=1)==48) # events with a label sum of 48 have the correct particles
            
    # filter events
    jets = jets[training_event_filter]
    electrons = electrons[training_event_filter]
    muons = muons[training_event_filter]
    labels = labels[training_event_filter]
    even = even[training_event_filter]
    
    return jets, electrons, muons, labels, even


def get_training_set(jets, electrons, muons, labels):
    '''
    Get features for each of the 12 combinations per event and calculates their corresponding combination-level labels
    
    Args:
        jets: selected jets after training filter
        electrons: selected electrons after training filter
        muons: selected muons after training filter
        labels: jet-level labels output by training_filter
    
    Returns:
        features, labels (flattened to remove event level)
    '''
    
    permutations_dict, labels_dict = get_permutations_dict(4, include_labels=True)
    
    features, perm_counts = get_features(jets, electrons, muons, max_n_jets=4)
    
    #### calculate combination-level labels ####
    permutation_labels = np.array(labels_dict[4])
    
    # which combination does the truth label correspond to?
    which_combination = np.zeros(len(jets), dtype=int)
    # no correct matches
    which_anti_combination = np.zeros(labels.shape[0], dtype=int)
    for i in range(12):
        which_combination[(labels==permutation_labels[i,:]).all(1)] = i
        which_anti_combination[np.invert((labels==permutation_labels[i,:]).any(1))] = i

    # convert to combination-level truth label (-1, 0 or 1)
    which_combination = list(zip(range(len(jets),), which_combination))
    which_anti_combination = list(zip(range(labels.shape[0],), which_anti_combination))
    
    truth_labels = -1*np.ones((len(jets),12))
    for i,tpl in enumerate(which_combination):
        truth_labels[tpl]=1
    for i,tpl in enumerate(which_anti_combination):
        truth_labels[tpl]=0
        
    #### flatten to combinations (easy to unflatten since each event always has 12 combinations) ####
    labels = truth_labels.reshape((truth_labels.shape[0]*truth_labels.shape[1],1))
    
    return features, labels, which_combination

# for generating triton config text
def write_triton_config(
    model_name, 
    n_features, 
    backend_name = "fil", 
    model_type = "xgboost", 
    max_batch_size = 50000000, 
    predict_proba = "true"
):
    n_out = 1
    if predict_proba=="true":
        n_out = 2
        
    return (f"name: \"{model_name}\"\n" 
            + f"backend: \"{backend_name}\"\n"
            + f"max_batch_size: {max_batch_size}\n"
            + "input [\n"
            + " {\n"
            + "    name: \"input__0\"\n"
            + "    data_type: TYPE_FP32\n"
            + f"    dims: [ {n_features} ]\n"
            + " }\n"
            + "]\n"
            + "output [\n"
            + " {\n"
            + "    name: \"output__0\"\n"
            + "    data_type: TYPE_FP32\n"
            + f"    dims: [ {n_out} ]\n"
            + " }\n"
            + "]\n"
            + "instance_group [{ kind: KIND_GPU }]\n"
            + "parameters [\n"
            + "  {\n"
            + "    key: \"model_type\"\n"
            + "    value: { string_value: "
            + f"\"{model_type}\""
            + " }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"predict_proba\"\n"
            + "    value: { string_value: "
            + f"\"{predict_proba}\""
            + " }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"output_class\"\n"
            + "    value: { string_value: \"true\" }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"threshold\"\n"
            + "    value: { string_value: \"0.5\" }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"algo\"\n"
            + "    value: { string_value: \"ALGO_AUTO\" }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"storage_type\"\n"
            + "    value: { string_value: \"AUTO\" }\n"
            + "  },\n"
            + "  {\n"
            + "    key: \"blocks_per_sm\"\n"
            + "    value: { string_value: \"0\" }\n"
            + "  }\n"
            + "]\n"
            + "version_policy: { all { }}\n"
            + "dynamic_batching { }")

    
