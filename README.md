This repository contains the code to reproduce the experiments of the paper:

[NeuMiss networks: differential programming for supervised learning with missing values](https://proceedings.neurips.cc/paper/2020/file/42ae1544956fbe6e09242e6cd752444c-Paper.pdf)


**If you want to try NeuMiss, we advise you to look at the [NeuMiss_sota repository](https://github.com/marineLM/NeuMiss_sota), which provides an easy-to-use PyTorch module implemeting NeuMiss.**

The file **NeuMiss.yml** indicates the packages required as well as the
versions used in our experiments.

The methods used are implemented in the following files:
 * **neumannS0_mlp**: the NeuMiss network.
 * **mlp**: the feedforward neural network.
 * **estimators**: the other methods used.

 The files **ground_truth** and **amputation** contain the code for data
 simulation and the code for the Bayes predictors.

 To reproduce the experiments, use:
  * `python launch_simu_perf MCAR`
  * `python launch_simu_perf MAR_logistic`
  * `python launch_simu_perf gaussian_sm`
  * `python launch_simu_perf probit_sm`
  * `python launch_simu_depth_effect`
  * `python launch_simu_architecture`

These scripts save their results as csv files in the **results** foder. The
plots can be obtained from these **csv** files by running the **plots_xxx**
files.
