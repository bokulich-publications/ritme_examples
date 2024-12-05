from ritme.find_best_model_config import (
    _load_experiment_config,
    _load_phylogeny,
    _load_taxonomy,
    find_best_model_config,
)
from ritme.split_train_test import _load_data, split_train_test
from ritme.evaluate_tuned_models import evaluate_tuned_models

######## USER INPUTS ########
# set experiment configuration path
model_config_path = "u1_rf_config.json"

# define path to feature table, metadata, phylogeny, and taxonomy
path_to_ft = "../../data/u1_subramanian14/otu_table_subr14_rar.tsv"
path_to_md = "../../data/u1_subramanian14/md_subr14.tsv"
path_to_phylo = "../../data/u1_subramanian14/fasttree_tree_rooted_subr14.qza"
path_to_tax = "../../data/u1_subramanian14/taxonomy_subr14.qza"

# define train size
train_size = 0.8
######## END USER INPUTS #####

# load ritme experiment configuration
config = _load_experiment_config(model_config_path)

# Perform train-test split
md, ft = _load_data(path_to_md, path_to_ft)
print(md.shape, ft.shape)

train_val, test = split_train_test(
    md,
    ft,
    stratify_by_column=config["stratify_by_column"],
    feature_prefix=config["feature_prefix"],
    train_size=train_size,
    seed=config["seed_data"],
)


# ## Find and evaluate optimal feature and model configuration with ritme
# find best model config
tax = _load_taxonomy(path_to_tax)
phylo = _load_phylogeny(path_to_phylo)

best_model_dict, path_to_exp = find_best_model_config(
    config, train_val, tax, phylo, path_store_model_logs="u1_rf_best_model"
)

# ## Evaluate feature and model configuration used by original paper
metrics = evaluate_tuned_models(best_model_dict, config, train_val, test)
metrics
