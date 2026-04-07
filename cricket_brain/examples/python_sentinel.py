"""Python usage example for cricket_brain PyO3 bindings."""

import cricket_brain

cfg = cricket_brain.BrainConfig()
cfg.seed = 1337
cfg.adaptive_sensitivity = True

brain = cricket_brain.Brain(cfg)

samples = [4500.0, 4500.0, 0.0, 4500.0, 0.0, 0.0]
outputs = [brain.step(x) for x in samples]

print("time_step:", brain.time_step)
print("outputs:", outputs)
