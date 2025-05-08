from training.src.utils.metrics import compute_homography_precision, compute_pose_errors, compute_symmetrical_epipolar_errors


def compute_step_metrics(batch, config):
    compute_symmetrical_epipolar_errors(
        batch
    )  # compute epi_errs for each match
    compute_pose_errors(
        batch, config
    )  # compute R_errs, t_errs, pose_errs for each pair
    homography_precision_thr = [3, 5, 10]
    compute_homography_precision(batch, homography_precision_thr)
    # compute_coarse_error(batch)

    rel_pair_names = list(zip(*batch["pair_names"]))
    bs = batch["image0"].size(0)
    metrics = {
        # to filter duplicate pairs caused by DistributedSampler
        "identifiers": ["#".join(rel_pair_names[b]) for b in range(bs)],
        "epi_errs": [
            batch["epi_errs"][batch["m_bids"] == b].cpu().numpy()
            for b in range(bs)
        ],
        "R_errs": batch["R_errs"],
        "t_errs": batch["t_errs"],
        "inliers": batch["inliers"],
        "elapsed_time": [batch["elapsed_time"]],
        "match_count": [batch["mkpts0_f"].shape[0]]
    }
    for thr in homography_precision_thr:
        metrics[f"H_auc@{thr}px"] = [batch[f"H_auc@{thr}px"]]
    
    ret_dict = {"metrics": metrics}
    return ret_dict, rel_pair_names