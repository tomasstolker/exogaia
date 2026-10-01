import pickle

import pytest

from exogaia.results import SamplingResults


def test_sampling_results_rejects_incomplete_pickle(tmp_path) -> None:
    pickle_file = tmp_path / "results.pkl"
    with pickle_file.open("wb") as open_file:
        pickle.dump({"samples": []}, open_file)

    with pytest.raises(KeyError, match="sampling results are missing"):
        SamplingResults(str(pickle_file))
