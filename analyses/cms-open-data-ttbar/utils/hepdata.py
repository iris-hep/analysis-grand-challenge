from hepdata_lib import Submission, Table, Variable, Uncertainty

def submission_hep_data(model, model_prediction, path):
    submission = Submission()
    for i in range(1, len(model.config.channels) +1):
        table = create_hep_data_table(i, model, model_prediction)
        submission.add_table(table)    
        submission.add_additional_resource("Workspace file", "workspace.json", copy_file=True) 
        submission.create_files(path, remove_old=True)
    
    

def create_hep_data_table(index, model, model_prediction):
    output = {}

    for i_chan, channel in enumerate(model.config.channels):
        for j_sam, sample in enumerate(model.config.samples):
            yields = model_prediction.model_yields[i_chan][j_sam]
            uncertainties = model_prediction.total_stdev_model_bins[i_chan][j_sam]
            num_bins = len(yields)
            for k_bin in range(num_bins):
                key = f"{channel} {sample} bin{k_bin}"
                value = {"value": yields[k_bin], "symerror": uncertainties[k_bin]}
                output[key] = value

    independent_variables = []
    dependent_variables = []
    independent_variables_ml = []
    dependent_variables_ml = []

    for key in output.keys():
        columns = key.split()
        if f'4j{index}b' in key:
            independent_variables.append(f"{columns[0]} {columns[1]} {columns[-1]}")
            dependent_variables.append(' '.join(columns[2:-1]))
        elif f'Feature{index}' in key:
            independent_variables_ml.append(f"{columns[0]} {columns[-1]}")
            dependent_variables_ml.append(' '.join(columns[1:-1])) 

    table_name = ""
    if independent_variables:
        table_name = f"4j{index}b Figure"
    elif independent_variables_ml:
        table_name = f"Feature{index} Figure"

    table = Table(table_name)

    var = Variable("sample", is_independent=True, is_binned=False, units="should be some units for the samples??")
    var.values = sorted(set(independent_variables + independent_variables_ml), key=lambda x: independent_variables.index(x) if x in independent_variables else independent_variables_ml.index(x))
    table.add_variable(var)
    for i, info in enumerate(model.config.samples):
        data_var = Variable(model.config.samples[i], is_independent=False, is_binned=False, units="Number of jets")
        data_var.values = model_prediction.model_yields[index-1][i]

        unc = Uncertainty("A symmetric error", is_symmetric=True)
        unc.values = model_prediction.total_stdev_model_bins[index-1][i]

        data_var.add_uncertainty(unc)
        table.add_variable(data_var)

    return table
