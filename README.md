# Value of Information Analysis for rationalising information gathering in building energy analysis

This repository supports the conference paper submission 'Value of Information Analysis for rationalising information gathering in building energy analysis', submitted to 'The 18th International IBPSA Conference and Exhibition - [Building Simulation 2023](https://bs2023.org/index)'. It provides the code used to perform the Value of Information (VoI) computations for the example problems presented in the submission.

## Technical Requirements

Use of this codebase requires Python 3.9 or later.

```
conda create --name myenv python>=3.9
pip install -r requirements.txt
```

## Codebase Structure

This repository contains:

- Three scripts for performing the EVPI calculations for each example problem:
    1. `building_ventilation.py`
    2. `ASHP_maintenance.py`
    3. `GSHP_design.py`
- `voi.py`, a generic implementation of the EVPI & EVII calculations for a general one-stage Bayesian decision problem.
- The `data` directory, containing input data and cached utility evaluations for the GHSP design example.
- The `plots` directory, containing code to produce the influence diagram figures for each example problem.
- A helpful caching function wrapper in `utils.caching`

## Note

The file `models/EP_Lamarche.py` has been removed from the repository due to copyright considerations. The authors will replace its functionality with an equivalent implementation that is openly available, however in the meantime `GSHP_design.py` can be run using the pre-evaluated results of the LaMarche model stored in `data/caches/`.