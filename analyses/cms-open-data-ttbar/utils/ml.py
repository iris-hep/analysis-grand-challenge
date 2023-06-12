import awkward as ak
import numpy as np
from xgboost import XGBClassifier
from .config import config

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

    # combined mass of b_toplep and lepton
    features[:, 4] = ak.flatten((leptons + jets[perms[..., 3]]).mass).to_numpy()

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
        results[np.invert(even)] = results_odd = model_even.predict_proba(
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