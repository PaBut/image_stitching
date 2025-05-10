import os
import pdb
import pprint
import subprocess

# import time
from collections import defaultdict
from pathlib import Path
import time
from typing import Literal

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from einops.einops import rearrange
from loguru import logger

from pipeline.enums import EnvironmentType
from pipeline.Modules.match_finders import AdaMatcherMatchFinder, FeatureDetector, FeatureDetectorMatchFinder, LoFTRMatchFinder
from AdaMatcher.src.lightning.lightning_common import compute_step_metrics
from AdaMatcher.src.utils.metrics import aggregate_metrics
from AdaMatcher.src.utils.misc import flattenList, lower_config
from AdaMatcher.src.utils.profiler import PassThroughProfiler

# from matplotlib import pyplot as plt

class PL_Tester(pl.LightningModule):
    def __init__(self, config, matcher_type: Literal['sift', 'adamatcher', 'loftr'], pretrained_ckpt=None, profiler=None, dump_dir=None):
        """
        TODO:
            - use the new version of PL logging API.
        """
        super().__init__()
        # Misc
        self.config = config  # full config
        self.profiler = profiler or PassThroughProfiler()

        torch.serialization.add_safe_globals([ModelCheckpoint])

        # Matcher: AdaMatcher
        if matcher_type == "sift":
            self.matcher = FeatureDetectorMatchFinder(FeatureDetector.SIFT)
        elif matcher_type == "loftr":
            self.matcher = LoFTRMatchFinder(EnvironmentType.Outdoor, pretrained_ckpt)
        elif matcher_type == "adamatcher":
            self.matcher = AdaMatcherMatchFinder(pretrained_ckpt)
        else:
            raise ValueError(f"Unknown matcher: {matcher_type}")

        torch.set_float32_matmul_precision('medium')

        # Testing
        self.dump_dir = dump_dir
        self.metric_time = 0.0

        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):

        img0 = batch["image0"].permute(0, 2, 3, 1).cpu().numpy() * 255
        img1 = batch["image1"].permute(0, 2, 3, 1).cpu().numpy() * 255
        start = time.perf_counter()
        k1, k2 = self.matcher.find_matches(img0[0], img1[0])
        end = time.perf_counter()
        logger.info(f"keypoints shape: {k1.shape}, {k2.shape}")
        batch["mkpts0_f"] = torch.from_numpy(k1).cuda().float()
        batch["mkpts1_f"] = torch.from_numpy(k2).cuda().float()
        batch["m_bids"] = torch.zeros(k1.shape[0], dtype=torch.long)
        batch["elapsed_time"] = (end - start) * 1000

        with self.profiler.profile("Compute metrics"):
            ret_dict, rel_pair_names = compute_step_metrics(batch, self.config)

        with self.profiler.profile("dump_results"):
            if self.dump_dir is not None:

                keys_to_save = {"mkpts0_f", "mkpts1_f", "epi_errs"}
                pair_names = list(zip(*batch["pair_names"]))
                bs = batch["image0"].shape[0]
                dumps = []
                for b_id in range(bs):
                    item = {}
                    mask = batch["m_bids"] == b_id
                    item["pair_names"] = pair_names[b_id]
                    item["identifier"] = "#".join(rel_pair_names[b_id])
                    for key in keys_to_save:
                        if "classification" not in key:
                            item[key] = batch[key][mask].cpu().numpy()
                        else:
                            item[key] = batch[key][b_id].cpu().numpy()
                    for key in [
                        "R_errs",
                        "t_errs",
                        "inliers",
                    ]:  # 'fp_scores', 'miss_scores']:
                        item[key] = batch[key][b_id]
                    dumps.append(item)
                ret_dict["dumps"] = dumps

        self.test_step_outputs.append(ret_dict)

        return ret_dict

    def on_test_epoch_end(self):
        # metrics: dict of list, numpy
        outputs = self.test_step_outputs
        _metrics = [o["metrics"] for o in outputs]
        metrics = {
            k: flattenList(([_me[k] for _me in _metrics]))
            for k in _metrics[0]
        }

        # [{key: [{...}, *#bs]}, *#batch]
        if self.dump_dir is not None:
            Path(self.dump_dir).mkdir(parents=True, exist_ok=True)
            _dumps = flattenList([o["dumps"] for o in outputs])  # [{...}, #bs*#batch]
            dumps = _dumps  # [{...}, #proc*#bs*#batch]
            logger.info(
                f"Prediction and evaluation results will be saved to: {self.dump_dir}"
            )

        if self.trainer.global_rank == 0:
            print(self.profiler.summary())
            val_metrics_4tb = aggregate_metrics(
                metrics, self.config.TRAINER.EPI_ERR_THR
            )
            logger.info("\n" + pprint.pformat(val_metrics_4tb))
            if self.dump_dir is not None:
                np.save(Path(self.dump_dir) / "Ada_pred_eval", dumps)

        self.test_step_outputs.clear()
