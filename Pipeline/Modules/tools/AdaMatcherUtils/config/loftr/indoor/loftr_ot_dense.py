from pipeline.Modules.tools.AdaMatcherUtils.config.default import _CN as cfg

cfg.ADAMATCHER.MATCH_COARSE.MATCH_TYPE = 'sinkhorn'

cfg.ADAMATCHER.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.MSLR_MILESTONES = [3, 6, 9, 12, 17, 20, 23, 26, 29]
