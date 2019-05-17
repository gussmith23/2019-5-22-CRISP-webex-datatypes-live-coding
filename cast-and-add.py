import tvm
# TVM operator index, containing things like topi.add
import topi

# Our basic program: Z = X + Y
X = tvm.placeholder((3, ))
Y = tvm.placeholder((3, ))
Z = topi.add(X, Y)

# Compile for LLVM (schedule, lower, and build)
target = "llvm"
schedule = tvm.create_schedule([Z.op])
lowered_func = tvm.lower(schedule, [X, Y, Z])
built_program = tvm.build(lowered_func, target=target)
