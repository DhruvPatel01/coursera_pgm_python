# Notes: 

- `I` will be a dict with three keys. 
    - random_factors
    - decision_factors
    - utility_factors
    
    Each of this valeus **MUST** be a list, even if the lenght of the list is one.

- Values for each of these keys will be a list of `Factor`s.

- Unlike the original MATLAB code, `print_factor` is not implemented separately.
    But `repr` is implemented on Factor, you can print them directly.

- `variable_elimination` is provided in `helper.py`

- Do copy `TestCases.m` into `data`, loading and testing samples is provided in `check.ipynb`.