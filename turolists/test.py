from __future__ import absolute_import, print_function

import tvm
import tvm.testing
from tvm import te
import numpy as np
import tvm.relay as relay
from tvm.contrib import graph_executor

# Global declarations of environment.

tgt_host = "llvm"
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl, rocm
tgt = "cuda"


n = te.var("n")
A = te.placeholder((n,), name="A")
B = te.placeholder((n,), name="B")
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")
s = te.create_schedule([C.op])
print(tvm.lower(s, [A, B, C], simple_mode=True)) # 可以查看schedule result