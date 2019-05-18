import tvm
# TVM operator index, containing things like topi.add
import topi
import numpy as np

# Our basic program: Z = X + Y
X = tvm.placeholder((3, ))
Y = tvm.placeholder((3, ))
Z = topi.add(X, Y)

# Compile for LLVM (schedule, lower, and build)
target = "llvm"
schedule = tvm.create_schedule([Z.op])
lowered_func = tvm.lower(schedule, [X, Y, Z])
built_program = tvm.build(lowered_func, target=target)

# Create a device context
context = tvm.context(target, 0)
# Create random input arrays on the above context
x = tvm.nd.array(np.random.rand(3).astype("float32"), ctx=context)
y = tvm.nd.array(np.random.rand(3).astype("float32"), ctx=context)
print("x:\t\t\t\t{}".format(x))
print("y:\t\t\t\t{}".format(y))

# This will hold our output.
z = tvm.nd.empty(Z.shape, dtype=Z.dtype, ctx=context)
