import torch

test_t5_P = torch.load('../dataset/test set/clathrin_ind_T5.pt')
test_t5_N = torch.load('../dataset/test set/vesicular_ind_T5.pt')
test_t5_P = torch.tensor(test_t5_P)  # list -> Tensor
test_t5_N = torch.tensor(test_t5_N)  # list -> Tensor
test_t5 = torch.cat((test_t5_P, test_t5_N), 0)


test_Bert_P = torch.load('../dataset/test set/clathrin_ind_Bert.pt')
test_Bert_N = torch.load('../dataset/test set/vesicular_ind_Bert.pt')
test_Bert = torch.cat((test_Bert_P, test_Bert_N), 0)


test_ESM3_P = torch.load('../dataset/test set/clathrin_ind_esm3.pt')
test_ESM3_N = torch.load('../dataset/test set/vesicular_ind_esm3.pt')
test_ESM3_P = torch.tensor(test_ESM3_P).float()
test_ESM3_N = torch.tensor(test_ESM3_N).float()
test_ESM3 = torch.cat((test_ESM3_P, test_ESM3_N), 0)
test_y = [1] * 258 + [0] * 227

