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

# Run the program.
built_program(x, y, z)
print("z:\t\t\t\t{}".format(z))

# Now, we will do the same, but we will use a custom datatype for our
# intermediate computation.

# The basic program, but with casts to a custom datatype.
# Note how we specify the custom datatype: we indicate it using the special
# `custom[...]` syntax.
# Additionally, note the "16" after the datatype: this is the bitwidth of the
# custom datatype. This tells TVM that each instance of bfloat is 16 bits wide.
Z = topi.cast(
    topi.cast(X, dtype="custom[bfloat]16") +
    topi.cast(Y, dtype="custom[bfloat]16"),
    dtype="float32")

# Initially, this will error out, saying:
# "TVMError: Check failed: name_to_code_.find(type_name) != name_to_code_.end():
#  Type name bfloat not registered"
# Thus, we need to register the datatype first!
