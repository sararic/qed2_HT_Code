# qed2_HT_Code
This code accompanies my master's thesis, “Hamiltonian Truncation Methods in Two-dimensional Quantum Electrodynamics.”

The following python packages are required: `scipy`, `numpy`, `numba`, `matplotlib`.

The code is documented with docstrings. The way I recommend it to be used is through a python3 interpreter, such as `ipython3`.

There are a few scripts that can be run with command-line arguments, but most of the useful features come when importing modules
such as `truncation_2p` or `2DQED_2p`, and using the functions and classes defined there.

If you press TAB twice after typing a module name and a dot in `ipython3`, it will show the various functions and classes that are available.
You can consult the docstring by typing `mymodule.myfunc?` in `ipython3`, or `help(mymodule.myfunc)` in plain python.
