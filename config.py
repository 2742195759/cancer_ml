y_label = 'year_1' # year_1 | year_3 | year_5

# x_labels = None
x_labels = [
    "stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6",
    "TMB",
    "ADGRB3",
    "DCDC1",
    "TTN",
    "IL5RA",
    "MIA",
    "P2RX1",
    "CD1E",
    "ADAMTS8",
    "NAPSA",
    "T_cells_CD8",
    "Macrophages_M2",
    "Dendritic_cells_resting",
]

test_set = "test" # "train" | "test"

default_random_seed = 54

#interested_classifier = 'LogisticRegression'
interested_classifier = 'Lasso'
#interested_classifier = None

interested_key = 'AUC'
