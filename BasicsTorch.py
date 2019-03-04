import torch

# Creating various tensors in Pytorch
torch_tensor_empty = torch.empty(3,3)
print('Empty Tensor:\n',torch_tensor_empty)

torch_tensor_random = torch.rand(3,3)
print('Random Valued Tensor:\n',torch_tensor_random)

torch_tensor_zeros = torch.zeros(3,3)
print('Zero Valued Tensor:\n',torch_tensor_zeros)

torch_tensor_custom = torch.tensor([[1,1,1],[2,2,2],[3,3,3]])
print('Custom Tensor:\n',torch_tensor_custom)

# Get shape and total elements
print('Shape of the custom tensor: ',torch_tensor_custom.size())
print('Total size of the custom tensor: ',torch_tensor_custom.numel())


# Addition of two tensors
torch_tensor_ones = torch.ones(3,3)
torch_tensor_twos = 2*torch.ones(3,3)

print('Tensor Addition Result by operator method:\n',torch_tensor_ones+torch_tensor_twos)
print('Tensor Addition Result by inbuilt function:\n',torch.add(torch_tensor_ones,torch_tensor_twos))

# Reshape a tensor
torch_tensor_ones = torch.ones(3,4)
torch_tensor_reshape_flatten = torch_tensor_ones.view(torch_tensor_ones.numel())
torch_tensor_reshape_oneDimensionFixed = torch_tensor_ones.view(-1,2)
print('Tensor Reshaping Result for flattening:\n',torch_tensor_reshape_flatten)
print('Tensor Reshaping Result for one dimension fixed:\n',torch_tensor_reshape_oneDimensionFixed)




