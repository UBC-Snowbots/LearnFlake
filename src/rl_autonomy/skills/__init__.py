# skills/ — modular skill training pipeline
#
# Each skill has three training stages:
#   train_coarse.py      — initial rough policy with relaxed tolerances
#   train_fine.py        — precision refinement with tighter tolerances
#   train_domain_rand.py — domain randomization for sim-to-real transfer
