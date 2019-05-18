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

# Register the custom datatype with tvm.
# Here, we register bfloats with a type code of 129. Currently, the type codes
# are chosen purely manually.
tvm.datatype.register("bfloat", 129)

# The basic program, but with casts to a custom datatype.
# Note how we specify the custom datatype: we indicate it using the special
# `custom[...]` syntax.
# Additionally, note the "16" after the datatype: this is the bitwidth of the
# custom datatype. This tells TVM that each instance of bfloat is 16 bits wide.
Z = topi.cast(
    topi.cast(X, dtype="custom[bfloat]16") +
    topi.cast(Y, dtype="custom[bfloat]16"),
    dtype="float32")

# Compile for LLVM (schedule, lower, and build)
schedule = tvm.create_schedule([Z.op])
lowered_func = tvm.lower(schedule, [X, Y, Z])
# Here, we manually lower custom datatypes. Soon, this will be incorporated
# directly into the TVM lower and build process.
lowered_func = tvm.ir_pass.LowerCustomDatatypes(lowered_func, target)
built_program = tvm.build(lowered_func, target=target)

# This will cause the following error:
# "TVMError: Check failed: lower: Cast lowering function for target llvm
#  destination type 129 source type 2 not found"
# This means that the custom datatype lowerer does not know how to lower Casts
# from float (type 2) to bfloat (type 129). So we need to tell it how!
