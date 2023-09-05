import cabinetry
from cabinetry.contrib import histogram_reader
import hist


def get_cabinetry_rebinning_router(config, rebinning):
    # perform re-binning in cabinetry by providing a custom function reading histograms
    # will eventually be replaced via https://github.com/scikit-hep/cabinetry/issues/412
    rebinning_router = cabinetry.route.Router()

    # this reimplements some of cabinetry.templates.collect
    general_path = config["General"]["InputPath"]
    variation_path = config["General"].get("VariationPath", None)

    # define a custom template builder function that is executed for data samples
    @rebinning_router.register_template_builder()
    def build_data_hist(region, sample, systematic, template):
        # get path to histogram
        histo_path = cabinetry.templates.collector._histo_path(general_path, variation_path, region, sample, systematic, template)
        h = hist.Hist(histogram_reader.with_uproot(histo_path))  # turn from boost-histogram into hist
        # perform re-binning
        h = h[rebinning]
        return h

    return rebinning_router
