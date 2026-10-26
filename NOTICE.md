# NOTICE: data sources and the terms that apply to the derived files

This repository releases code (MIT licence, `LICENSE`), analysis tables and, as release assets, trained
checkpoints. The data sets themselves are not redistributed. The event files of the study were derived from the
sources below, and every file derived from them, the trained checkpoints included, remains subject to the terms of
its source. The code is MIT-licensed; the terms below are those of the data providers, not of this repository.

| Source | Used as | Terms of the provider |
|---|---|---|
| highD (Krajewski et al., 2018; levelXdata) | the highD car-following events as released with the FollowNet benchmark (Chen et al., 2023); the raw highD recordings were not used | highD is free for non-commercial use only: the data set and modified versions may not be redistributed, derivative works may be distributed only as abstract representations of the data set, and no derivative work may be used commercially (levelXdata terms of use). The checkpoints trained on highD-derived events are such abstract representations and are provided for non-commercial research use only. |
| NGSIM I-80 and US-101 (U.S. Department of Transportation, ITS DataHub, doi:10.21949/1504477) | the FollowNet NGSIM I-80 events and the US-101 reconstruction of the raw trajectories | Published by the U.S. Department of Transportation; the catalogue record names the Creative Commons Attribution-ShareAlike 3.0 Unported licence. Attribution: U.S. Department of Transportation Intelligent Transportation Systems Joint Program Office (JPO). |
| Waymo Open Dataset (Sun et al., 2020), car-following pairs processed by Hu et al. (2021, 2022) | the `waymo` event set, and the checkpoints of every root that trained on it (see below) | Provided by Waymo LLC under the Waymo Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms). Under that agreement a machine-learning model trained on the data set is Derivative IP: it may be published for non-commercial purposes only, it must carry a notice that the agreement applies, and it may not be used in the operation of a vehicle or to assist in operating one. The Waymo-trained checkpoints are released under these conditions, with this notice; recipients obtain the agreement at the link above. |
| openACC (Makridis et al., 2021, doi:10.1016/j.trc.2021.103047; Joint Research Centre of the European Commission) | the `openacc` event set and the platoon leader profiles of `configs/platoon/profiles*.yaml` | Open-access database published by the JRC; reuse under the terms of its JRC Data Catalogue record, with attribution to the authors. |

Which checkpoints trained on which data: the `config.yaml` of every experiment root names its training sets.
The roots trained on highD only (the E1 roots and their one-step controls, the NGSIM-trained root of the
extension excepted) are subject to the highD terms; the pooled-domain roots (`e6_pool`, `e6_lodo_ngsim`,
`e6_lodo_ngsim_all`, `e6_lodo_openacc`, their history-only controls `e6_pool_ctrl` and `e6_lodo_openacc_ctrl`, the
budget-trained roots `ext_pool_prev1`, `ext_pool_prev2`, `ext_lodo_prev1`, `ext_lodo_prev2` and the closed-loop
fine-tuned roots `ext_cl_pool` and `ext_cl_lodo`) trained on the Waymo pairs as well and are subject to the highD
and the Waymo terms together. The NGSIM and openACC sources add the attribution requirements above.

Attribution line for the Waymo Open Dataset, as its agreement asks: "The Waymo Open Dataset is provided by Waymo
LLC under the Waymo Dataset License Agreement for Non-Commercial Use (https://waymo.com/open/terms)."
