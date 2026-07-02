"""
postulate2 -- reference implementation of the Recursive Scale Structure upgrade.

Public modules:
    engine       the Postulate-1 correction engine + two guarded hooks
    environment  the two-scale toy task (hidden 2D manifold -> 20D observation)
    scale1       Scale-1 representation learner (crystallize particles -> config space)
    encoders     encoders over the emergent config space (incl. interactive)
    scale2       run the engine on any representation; metrics
    promotion    basin detection + context-aware interface promotion
    validate     end-to-end functional validation (run: python postulate2/validate.py)

The modules import each other by bare name, so run scripts from inside this
directory (or add it to sys.path) -- see README.md.
"""
