"""TopoNull with 32 instead of 8 permutations (Monte Carlo convergence check). NAME znull32."""
import feat_null as FN

NAME = "znull32"


def compute(im, n_types):
    k = FN.K_NULL
    FN.K_NULL = 32
    try:
        return FN.compute(im, n_types)
    finally:
        FN.K_NULL = k
