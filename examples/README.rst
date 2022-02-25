.. role:: raw-html-m2r(raw)
   :format: html

Basic Usage
-----------
This is a simple example for using `crabnet`.

.. literalinclude:: ../../examples/crabnet_basic.py
    :caption: `/examples/crabnet_basic.py <https://github.com/sparks-baird/crabnet/blob/main/examples/crabnet_basic.py>`_
    :language: python

Extend Features Comparison
--------------------------
This is a comparison of a recently (Feb 2022) implemented extend features functionality
with XGBoost using a hardness dataset consisting of approximately 500 unique
compositions across 1000 entries.

.. literalinclude:: ../../examples/extend_features_compare.py
    :caption: `/examples/extend_features_compare.py <https://github.com/sparks-baird/crabnet/blob/main/examples/extend_features_compare.py>`_
    :language: python
    
Bare-bones teaching example
---------------------------
If you're interested in learning more about the CrabNet architecture, but with fewer enhancements, see the following example.

``crabnet_teaching.py`` takes care of the top-level PyTorch model architecture (i.e.
``fit`` and ``predict``). A minimal ``SubCrab`` transformer model is defined in
``subcrab_teaching.py``. You may find it useful to use an IDE with a debugger (e.g. VS
Code, PyCharm, or Spyder) to step through the code.

.. literalinclude:: ../../examples/crabnet_teaching.py
    :caption: `/examples/crabnet_teaching.py <https://github.com/sparks-baird/crabnet/blob/main/examples/crabnet_teaching.py>`_
    :language: python

.. literalinclude:: ../../examples/subcrab_teaching.py
    :caption: `/examples/subcrab_teaching.py <https://github.com/sparks-baird/crabnet/blob/main/examples/subcrab_teaching.py>`_
    :language: python