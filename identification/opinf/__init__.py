# opinf/__init__.py
"""Operator Inference for data-driven model reduction of dynamical systems.
"""

__version__ = "0.0.1"

# The order matters. parameters, errors and utils are the foundation: several
# modules below call opinf.parameters.Params() or opinf.utils.get_device() at
# import time, so those three have to be bound on the package before anything
# that does. models in particular pulls in training, which reaches for
# opinf.utils while opinf itself is still executing.
#
# Until recently this worked by accident: basis/_pod.py imported opinf.post,
# which imported ..utils, which bound utils early enough. That import was
# unused and has been removed, which is why the order now has to be stated.
from . import parameters, errors, utils
from . import basis, models, post, pre, training
