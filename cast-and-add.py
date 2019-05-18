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

# Register lowering functions. The first argument of register_op is a function
# whichperforms the lowering for this specific op. That is, it takes an op (e.g.
# a Cast) and returns a new op which is equivalent, but does not use the custom
# datatype.
# We provide a convenience function, create_lower_func(), which simply lowers
# the op to a Call to a function of the provided name, e.g. BFloat16Add_wrapper.
tvm.datatype.register_op(tvm.datatype.create_lower_func("FloatToBFloat16_wrapper"),
                         "Cast", target, "bfloat", "float")
tvm.datatype.register_op(tvm.datatype.create_lower_func("BFloat16ToFloat_wrapper"),
                         "Cast", target, "float", "bfloat")
tvm.datatype.register_op(tvm.datatype.create_lower_func("BFloat16Add_wrapper"),
                         "Add", target, "bfloat")


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


# Finally, create a new array to hold the output and run the program.
z_bfloat = tvm.nd.empty(Z.shape, dtype=Z.dtype, ctx=context)

built_program(x, y, z_bfloat)
print("z_bfloat:\t{}".format(z_bfloat))
