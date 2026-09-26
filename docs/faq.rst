.. _faq:

Frequently Asked Questions
==========================

What dependency versions should I use?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``exogaia`` package is kept reasonably well up to date with recent dependency versions. If you encounter an error then try to update the dependencies.

For example, in your local folder where you may have cloned the repository:

.. code-block:: bash

   pip install --upgrade -e .

How do I run my code on multiple CPUs?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The orbit inference with ``NestedSampler`` supports multiprocessing.

First, make sure to install ``mpi4py``:

.. code-block:: bash

   pip install mpi4py

Then, to execute you ``exogaia`` script with MPI, for example using 8 CPUs:

.. code-block:: bash

   mpirun -n 8 python run_exogaia.py
