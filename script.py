import numpy as np
import os
import torch


train_nid, val_nid, test_nid, num_nodes, labels = torch.load('data_splits.pt')

err1, err2 = [], []
for filename in os.listdir('./output/icdm22'):
	if filename.endswith('.pt') and 'test' not in filename:
		raw_preds = torch.load(f'output/icdm22/{filename}')
		train_err_mask = raw_preds[:len(train_nid)].argmax(1) != labels[train_nid]
		train_err = train_err_mask.sum().item()
		val_err_mask = raw_preds[len(train_nid):len(train_nid)+len(val_nid)].argmax(1) != labels[val_nid]
		val_err = (val_err_mask).sum().item()
		print(filename, train_err, val_err)

		if val_err < 500 and val_err > 200:
			a = torch.where(train_err_mask)[0]
			b = torch.where(val_err_mask)[0]
			err1.append(a)
			err2.append(b)

train_err_times = torch.zeros(num_nodes, dtype=torch.long)
val_err_times = torch.zeros(num_nodes, dtype=torch.long)

for ele in err1:
	train_err_times[ele] += 1
print(torch.unique(train_err_times, return_counts=True))
# 1: 318
# 2: 190
# 3: 108
# 4: 36
# 5: 109

for ele in err2:
	val_err_times[ele] += 1
print(torch.unique(val_err_times, return_counts=True))
# 1: 79
# 2: 63
# 3: 55
# 4: 59
# 5: 226

# torch.save((train_err_times, val_err_times), 'err_times.pt')

train_err_nodes = torch.where(train_err_times)[0]
train_err_times = train_err_times[train_err_nodes]
val_err_nodes = torch.where(val_err_times)[0]
val_err_times = val_err_times[val_err_nodes]
torch.save((train_err_nodes, train_err_times, val_err_nodes, val_err_times), 'err_times.pt')