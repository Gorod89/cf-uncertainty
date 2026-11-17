# NOTICE: data sources and the terms that apply to the derived files

This repository releases code (MIT licence, `LICENSE`), analysis tables and, as release assets, trained
checkpoints. Since release v1.5.3 the checkpoints are attached as one archive per group of data sources, each with its own notice and manifest, and the checkpoints of eleven roots are not distributed (see below). No highD, NGSIM or Waymo data are redistributed: neither trajectories nor the event files derived from
them are in the tree, its history or the release assets. Of openACC the repository contains the leader speed series
of the 60 platoon windows of the platoon experiment (`configs/platoon/profiles.yaml` and
`results/extension/*/profiles__*.yaml`, also inside the platoon-table archives of the releases): windows selected from
the raw files, at 10 Hz, smoothed with a Savitzky-Golay filter; the copyright notice delivered with openACC allows reuse provided
appropriate credit is given and any changes are indicated (see the table below). Every file derived from a source, the trained checkpoints included, remains subject to
the terms of that source. The code is MIT-licensed; the terms below are those of the data providers, not of this
repository.

This notice reports the terms of the providers as they were read on 16 November 2026. It does not merge the terms of
several sources into one licence, and it is not legal advice. A sentence that is a condition of this project, and not of a provider, is marked as such.

| Source | Used as | Terms of the provider |
|---|---|---|
| highD (Krajewski et al., 2018; Institute for Automotive Engineering of RWTH Aachen University, leveLXData) | the highD car-following events as released with the FollowNet benchmark (Chen et al., 2023); the raw highD recordings were not used | Terms and conditions at https://levelxdata.com/highd-dataset/: the data set is free for non-commercial use, on application. The data set and modified versions may not be distributed. Derivative works may be distributed as far as they are abstract representations of the data set, trained models being named as an example, that do not allow the data set, or something similar in character, to be recovered. Neither the data set nor a derivative work may be used for commercial purposes. Every work that uses the data set references it. The checkpoints trained on highD-derived events hold trained parameters only (see below); the fold definitions `data/splits/follownet_highd_driver.json` list the event numbers of the FollowNet release with their fold and hold no data of the data set. |
| NGSIM I-80 and US-101 (U.S. Department of Transportation, Federal Highway Administration; ITS DataHub, doi:10.21949/1504477) | the I-80 and US-101 event sets reconstructed from the raw trajectories of data.transportation.gov (the reconstruction of the companion project, https://github.com/Gorod89/cf-stability): evaluation sets of every root, training sets of the pooled-domain roots and of the NGSIM-trained root of the extension; the NGSIM I-80 events of the FollowNet benchmark as a cross-check of the evaluation only | The record of the data set on data.transportation.gov (id `8ect-6jqj`) names the licence "Creative Commons Attribution \| Share Alike 3.0 Unported" (licence id `CC_30_BY_SA`, https://creativecommons.org/licenses/by-sa/3.0/legalcode). No NGSIM trajectory or event file is redistributed here. The fold definitions `data/splits/ngsim_i80_driver.json` and `data/splits/ngsim_us101_driver.json` list the follower and leader identifiers of the events with their fold; like the other files computed from NGSIM (fitted parameters, tables, the checkpoints of `ext_e1_ngsim`) they hold no NGSIM trajectory and remain subject to the terms of this source, and this notice does not state their legal status under it. Changes made to the source for the event sets: reconstruction and smoothing of the trajectories, conversion to SI units, extraction of car-following events. |
| Waymo Open Dataset (Sun et al., 2020), car-following pairs processed by Hu et al. (2021, 2022) | the `waymo` event set, and the checkpoints of every root that trained on it (see below) | Provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms; the text read is dated March 2025). The agreement counts a model trained on the data set, its weights included, as Derivative IP. It allows trained weights to be published on these conditions: a prominent notice that the model is released under the agreement and that any further use or modification is subject to its terms, the non-commercial restrictions included; the agreement applies to the recipients for the whole of the model; the recipients are given a copy of the agreement; the attribution below. Such a model may not be used in the operation of a vehicle, in production systems or for primarily commercial purposes. The distributed checkpoints of the root that trained on the Waymo pairs (`e6_lodo_ngsim_all`), the parameters fitted on the `waymo` event set or on a pooled domain that includes it and kept in the tree (`data/calibration/waymo/`, the fit logs of the pooled roots under `results/`) and the rows of the tables computed from the pairs are released under that agreement, with this notice; a copy of the agreement is in `WAYMO_OPEN_DATASET_LICENSE.txt` and in the folder `licenses/` of the archive `checkpoints_highd_waymo_openacc.tar.gz`. The Mendeley record of the pairs carries a CC BY 4.0 label; the agreement covers any data of the data set, and this project applies it to everything computed from the pairs. The fold definition of the pairs (`data/splits/waymo_driver.json`), which lists the segment and vehicle labels of the pairs, is not distributed since release v1.5.4; it is rebuilt from the reader's own copy of the pairs (README, section Data). |
| openACC (European Commission, Joint Research Centre: Open ACC Database, doi:10.2905/JRC.KMH3D00; Makridis et al., 2021, doi:10.1016/j.trc.2021.103047) | the `openacc` event set (not redistributed) and the platoon leader profiles (redistributed as smoothed speed series, see above) | The copyright notice delivered with the data (`copyright.txt` in every campaign folder, © European Union) licenses any copyright and sui generis right on the data set under CC BY 4.0: reuse is allowed provided appropriate credit is given and any changes are indicated. The catalogue record names the same basis, the European Commission reuse notice (Commission Decision 2011/833/EU). The fold definitions `data/splits/openacc_driver.json` name vehicles and files of the campaigns and hold no series of the data set. |
| FollowNet (Chen et al., 2023; figshare collection doi:10.6084/m9.figshare.c.6777810) | source of the highD events and of the NGSIM I-80 cross-check events | figshare labels the event files CC0. That label does not replace the terms of highD and NGSIM, which this notice follows. |

## Which checkpoints were trained on which data

The `config.yaml` of every experiment root names its training sets (`train.data`).

| Experiment roots | Release archive (release v1.5.3) | Training sets |
|---|---|---|
| `e1`, `e1_ctrl`, `e1_mean1s`, `e1h`, `e1h_mean1s`, `e1m`, `e1m_causal`, `e1m_mean1s`, `e1m_mean1s_abv`, `e1m_rt`, `ext_cl_highd` | `checkpoints_highd.tar.gz` | highD |
| `ext_e1_ngsim` | `checkpoints_ngsim.tar.gz` | NGSIM I-80 |
| `e6_lodo_ngsim_all` | `checkpoints_highd_waymo_openacc.tar.gz` | highD, Waymo, openACC |
| `ext_idm_modes` (two calibrations of the intelligent driver model) | `checkpoints_openacc_idm.tar.gz` | openACC |
| `e6_pool`, `e6_pool_ctrl`, `ext_pool_prev1`, `ext_pool_prev2`, `ext_cl_pool` | not distributed | highD, NGSIM I-80, NGSIM US-101, Waymo, openACC |
| `e6_lodo_openacc`, `e6_lodo_openacc_ctrl`, `ext_lodo_prev1`, `ext_lodo_prev2`, `ext_cl_lodo` | not distributed | highD, NGSIM I-80, NGSIM US-101, Waymo |
| `e6_lodo_ngsim` | not distributed | highD, NGSIM US-101, Waymo, openACC |

The pooled-domain roots keep the intelligent driver model calibrated on highD as a fixed prior, so each of them
depends on highD even where its learned part saw other sets as well.

Every archive carries its own notice (`NOTICE_checkpoints_<group>.txt`) with the terms of its sources as they were
read, and a SHA-256 manifest of its files. The checkpoints of `e6_lodo_ngsim_all` were trained on highD-derived events,
the Waymo pairs and openACC events: they are released under the Waymo agreement, as described above, and remain subject
to the highD terms. The checkpoints of `ext_e1_ngsim` remain subject to the terms of the NGSIM record; this notice does
not state their legal status under it.

Condition of this project, following point 4 of the highD terms and Section 4 of the Waymo agreement: every file with
the highD events or the Waymo pairs in its provenance is provided for non-commercial research and teaching only.

The checkpoints of the eleven roots trained on NGSIM sets together with highD and Waymo data are not distributed. They
were attached to the releases v1.3.0 to v1.5.2 and were withdrawn on 17 November 2026: the NGSIM record carries a
ShareAlike licence, the highD and Waymo terms allow non-commercial use only, and the Waymo agreement asks to apply to a
published model as a whole. This notice does not state whether one set of conditions can meet all of these terms; until
that is settled the files are not published. The results obtained with these checkpoints stay in `results/`, and the
commands of the README regenerate the checkpoints from the released configurations (about 11 GPU hours).

## What the checkpoints hold

Trained parameters and normalisation constants, no rows of a data set. The one model class with stored points of the
input space is the Gaussian-process residual law: each of its checkpoints keeps 300 inducing inputs, points of the
standardised state space (gap, relative speed, speed) that start at cluster centres and are trained with the other
parameters. `extension/scripts/gp_inducing_check.py` compares them with the training rows of every root: none of the
21 000 inducing inputs of the 70 checkpoints lies within 1e-3 standardised units of a training row
(`results/extension/ext_gp_inducing/tables/`).

## Attributions

* highD: Krajewski, R.; Bock, J.; Kloeker, L.; Eckstein, L. The highD Dataset: A Drone Dataset of Naturalistic Vehicle
  Trajectories on German Highways for Validation of Highly Automated Driving Systems. In Proceedings of the 2018 21st
  International Conference on Intelligent Transportation Systems (ITSC), 2018; pp. 2118-2125.
  doi:10.1109/ITSC.2018.8569552.
* NGSIM: U.S. Department of Transportation Federal Highway Administration. (2016). Next Generation Simulation (NGSIM)
  Vehicle Trajectories and Supporting Data. [Dataset]. Provided by ITS DataHub through Data.transportation.gov.
  http://doi.org/10.21949/1504477.
* Waymo Open Dataset: the checkpoints of the root trained on the Waymo pairs, the parameters fitted on the pairs and the results obtained with them
  were made using the Waymo Open Dataset, provided by Waymo LLC under the Waymo Dataset License Agreement for
  Non-Commercial Use, available at waymo.com/open/terms, and your access and use of such work are governed by the
  terms and conditions therein. The car-following pairs: Hu, X.; Zheng, Z.; Zhang, X.; Chen, D.; Sun, J. Vehicle
  Trajectory Data Processed from the Waymo Open Dataset. Mendeley Data, Version 2, 2021. doi:10.17632/wfn2c3437n.2.
* openACC: Makridis, M.; Mattas, K.; Anesiadou, A.; Ciuffo, B. OpenACC. An open database of car-following experiments
  to study the properties of commercial ACC systems. Transportation Research Part C: Emerging Technologies 2021, 125,
  103047. doi:10.1016/j.trc.2021.103047. Data set record: doi:10.2905/JRC.KMH3D00.
* FollowNet: Chen, X.; Zhu, M.; Chen, K.; Wang, P.; Lu, H.; Zhong, H.; Han, X.; Wang, X.; Wang, Y. FollowNet: A
  Comprehensive Benchmark for Car-Following Behavior Modeling. Scientific Data 2023, 10, 828.
  doi:10.1038/s41597-023-02718-7.
