1. pytorch v0.3.1
2. -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DNNPACK_BACKEND="psimd"
3. cp libnnpack.a and libpthreadpool.a to NNPACK/lib
4. export NNPACK_DIR=
5. sudo -E python setup.py install
6. vision 0.2.1
7. Pytorch "ensure large enough batch size to ensure perf, tuneable" to enable NNPACK (convolution.cpp)

.py
import torchvision.models as models
import torch
from torch.autograd import Variable

resnet50 = models.resnet50()

input = torch.randn(1, 3, 224, 224)

output = resnet50(Variable(input))

print output