import tvm
# TVM operator index, containing things like topi.add
import topi

# Our basic program: Z = X + Y
X = tvm.placeholder((3, ))
Y = tvm.placeholder((3, ))
Z = topi.add(X, Y)
