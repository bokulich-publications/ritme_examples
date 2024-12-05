import os
import sys
from itertools import compress

import pandas as pd
import qiime2 as q2


def need2refetch(output_path):
    """
    Function that verifies whether `failed_runs.qza` artifact
    in `output_path` is non-empty and with this one needs to refetch
    some runIDs. If `failed_runs.qza` is empty it returns `False`,
    if not it returns `True`.
    """

    path2file = os.path.join(output_path, "failed_runs.qza")

    if not os.path.isfile(path2file):
        raise ValueError(f"{path2file} does not exist")
    else:
        df_failed_runs = q2.Artifact.load(path2file).view(pd.DataFrame)
        nb_failed = df_failed_runs.shape[0]

        if nb_failed > 0:
            return True
        else:
            return False


def get_refetch_directories(path2check):
    """
    Function that checks in each folder in `path2check`
    whether `failed_runs.qza` artifact is empty. It returns
    a list with all folder names that contain a non-empty
    `failed_runs.qza` artifact and need to be refetched.
    """

    # get all child directories, hence [1:]
    ls_all_dir = [x[0] for x in os.walk(path2check)][1:]
    ls_verify = []

    for dir in ls_all_dir:
        # inverting with Not as we want the folders where
        # failedids is NOT empty
        ls_verify.append(need2refetch(dir))

    assert len(ls_verify) == len(ls_all_dir)

    return list(compress(ls_all_dir, ls_verify))


if __name__ == "__main__":
    # print return value
    print(f"{need2refetch(sys.argv[1])}")
