# ASAP-AES data

Download `training_set_rel3.tsv` from the competition's Data tab and save it as `data/raw/training_set_rel3.tsv`. Kaggle may require signing in and accepting the competition rules. [ASAP-AES on Kaggle](https://www.kaggle.com/competitions/asap-aes/data)

The loader uses `essay_id`, `essay_set`, `essay`, and `domain1_score`. Each run selects one essay set and uses its Domain 1 score as the regression target. The labeled training file supplies the local train, validation, and test splits.

Keep the competition files in `data/raw/`. Git ignores downloaded data and generated outputs.
