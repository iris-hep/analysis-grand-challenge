import uproot
from hepdata_lib import Submission, Table, Variable, Histogram1D

file_path = "workspace.json"
root_file = uproot.open(file_path)

#Access the NTuple inside the ROOT file

ntuple = root_file["workspace.json"]

#Extract data from NTuple

data = ntuple["histogram_variable"].array()

#Create a hepdata_lib submission

submission = Submission()

table = Table("Table 1")

variable = Variable("X-axis", is_independent=True, units='GeV')
variable.values = data.bins[:-1]

#Create a histogram 
histogram = Histogram1D(
    "Histogram Name",
    data.counts,
    uncertainties=data.errors,
    label="Y-axis",
    units="Events",
)

table.add_variable(variable)
table.add_variable(histogram)

submission.add_table(table)

submission_path = "submission.yaml"
submission.to_yaml(submission_path)