# SeHGNN for ICDM22 contest

This repository is the implemention of SeHGNN on the [icdm 22 contest](https://tianchi.aliyun.com/competition/entrance/531976/introduction).

PS: We attend this contest for the aim of evaluting the effects of our SeHGNN on more datasets, but the results are not very satisfactory. Except for the inbalance between training and testing data, maybe this contest is more like a problem about anomaly detection rather than simple node classification. The utilization of edge attentions is one effective tool in defense of attacks on graphs, which the current SeHGNN does not contain.

## Environment setup

Please follow [SeHGNN](https://github.com/ICT-GIMLab/SeHGNN/).

## Data preparation

Please clone [icdm\_graph\_competition](https://git.openi.org.cn/GAMMALab/icdm_graph_competition) and set its path as `ICDM_ROOT`.

```setup
git clone https://git.openi.org.cn/GAMMALab/icdm_graph_competition.git --depth=1
cd icdm_graph_competition
ICDM_DIR=$(pwd)
mkdir -p data/dgl_data
```

Download necessary data from [the contest website](https://tianchi.aliyun.com/competition/entrance/531976/information) and put it under `$ICDM_DIR/data`, then run:

```setup
cd $ICDM_DIR/dgl_example
mkdir -p ../data/dgl_data/icdm2022_session1
python format_dgl.py --graph=../data/icdm2022_session1_edges.csv --node=../data/icdm2022_session1_nodes.csv --storefile=../data/dgl_data/icdm2022_session1/icdm2022_session1
mv ../data/icdm2022_session1.* ../data/dgl_data/icdm2022_session1
cp ../data/icdm2022_session1_train_labels.csv ../data/dgl_data/icdm2022_session1/icdm2022_session1_labels.csv
cp ../data/icdm2022_session1_test_ids.txt ../data/dgl_data/icdm2022_session1/icdm2022_session1_test_ids.csv
mkdir -p ../data/dgl_data/icdm2022_session2
python format_dgl.py --graph=../data/icdm2022_session2_edges.csv --node=../data/icdm2022_session2_nodes.csv --storefile=../data/dgl_data/icdm2022_session2/icdm2022_session2
mv ../data/icdm2022_session2.* ../data/dgl_data/icdm2022_session2
cp ../data/icdm2022_session2_test_ids.txt ../data/dgl_data/icdm2022_session2/icdm2022_session2_test_ids.csv
```

## Training

Please make sure the enviroment variable `ICDM_ROOT` has been correctly set.

To reproduce the results on the contest:

```setup
python main.py --dataset icdm22 --root $ICDM_DIR --num-hops 2 --n-layers-1 1 --n-layers-2 2 \
	--residual --act leaky_relu --lr 0.001 --weight-decay 0 --patience 50 --amp \
	--test-batch-size 32768 --batch-size 1024 --enlarge-val-set --run-times 1
```

PS: For the first time running, it would take extra time to generate temporary data files and save them to disk.

## Trick

We find around 15% positive nodes of the validation set cannot be correctly classified for almost all models. So we re-split the dataset by adding these nodes to the training set. We have provided positions of these nodes in the file `err_times.pt`.

To generate a new `err_times.pt` file, firstly run the original SeHGNN model without enlarging the validation set, and then list the times that each validation node is not correctly classified.

```setup
python main.py --dataset icdm22 --root $ICDM_DIR --num-hops 2 --n-layers-1 1 --n-layers-2 2 \
	--residual --act leaky_relu --lr 0.001 --weight-decay 0 --patience 30 --amp \
	--test-batch-size 32768 --batch-size 1024 --run-times 5
python script.py
```
