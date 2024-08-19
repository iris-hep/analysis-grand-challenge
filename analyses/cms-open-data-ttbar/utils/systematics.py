import awkward as ak
import numpy as np
import correctionlib.schemav2 as cs

# functions creating systematic variations
def jet_pt_resolution(pt,phi):
    """
    normal distribution with 5% variations, shape matches jets. Uses phi as a source
    of entropy in correctionlib.

    Inputs:
        pt: Awkward array of floats
            The pt of the jets to smear
        phi: Awkward array of floats
            The phis of the jets to smear. Just used as a source of entropy for
            correctionlib
    """
    res = cs.Correction(
        name="res",
        description="Deterministic smearing value generator",
        version=1,
        inputs=[
            cs.Variable(name="pt", type="real", description="Unsmeared jet pt"),
            cs.Variable(name="phi", type="real", description="Jet phi (entropy source)"),
        ],
        output=cs.Variable(name="factor", type="real"),
        data=cs.HashPRNG(
            nodetype="hashprng",
            inputs=["pt","phi"],
            distribution="stdnormal",
        )
    )

    vals = res.to_evaluator().evaluate(pt,phi)
    #Make the standard deviation 0.05
    vals = 0.05 * vals
    #Make the mean 1
    rtn = vals + 1
    return rtn