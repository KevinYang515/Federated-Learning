# Federated-Learning

This is a simple reproduction to partly implement and simulate the paper of [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629 "FedAvg").


## Example

    python fedavg_plus.py

If you want to change data distribution of each device, you can modify the file (i.e., data / file / data_distribution_3000_new_14.txt). And the type in that file will be like, emd; data distribution (from class 0 to 9); variance, if you want to adjust the data quantity for any class, you only have to modify the number of that class of the specific device.