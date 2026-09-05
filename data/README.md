# Data preparation

Datasets are not distributed with this repository. Pass a local file using `--data`.
MAT files require features under `data`, `X`, or `features`, and labels under `target`, `Y`, or `labels`.
NPZ files require `X` and `Y`. Features have shape N-by-D; labels should have shape N-by-L.
L-by-N labels are transposed when their first dimension differs from N. For square arrays, provide N-by-L explicitly.
Logical labels encoded as -1 are converted to 0. Missing values are rejected.
MATLAB v7.3/HDF5 and predefined train/test layouts are not supported by this loader.
The synthetic demo needs no download. See the manuscript dataset references for benchmark provenance and terms.
