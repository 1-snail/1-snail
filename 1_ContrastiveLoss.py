# %matplotlib inline
import logging

import matplotlib.pyplot as plt
import numpy as np
import record_keeper
import torch
import torch.nn as nn
import torchvision
# import umap
from cycler import cycler
from PIL import Image
from torchvision import datasets, transforms
import tensorboard
import pytorch_metric_learning
import pytorch_metric_learning.utils.logging_presets as logging_presets
from pytorch_metric_learning import losses, miners, samplers, testers, trainers
from pytorch_metric_learning.utils import common_functions
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from MyData import MyDataset,ClassDisjointMyDataset
import datetime
import json
import os

logging.getLogger().setLevel(logging.INFO)
logging.info("VERSION %s" % pytorch_metric_learning.__version__)

class MLP(nn.Module):
    # layer_sizes[0] is the dimension of the input
    # layer_sizes[-1] is the dimension of the output
    def __init__(self, layer_sizes, final_relu=False):
        super().__init__()
        layer_list = []
        layer_sizes = [int(x) for x in layer_sizes]
        num_layers = len(layer_sizes) - 1
        final_relu_layer = num_layers if final_relu else num_layers - 1
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            curr_size = layer_sizes[i + 1]
            if i < final_relu_layer:
                layer_list.append(nn.ReLU(inplace=False))
            layer_list.append(nn.Linear(input_size, curr_size))
        self.net = nn.Sequential(*layer_list)
        self.last_linear = self.net[-1]

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set trunk model and replace the softmax layer with an identity function
trunk = torchvision.models.resnet18(pretrained=True)
trunk_output_size = trunk.fc.in_features
trunk.fc = common_functions.Identity()
# trunk = torch.nn.DataParallel(trunk.to(device))
trunk = trunk.to(device)

# Set embedder model. This takes in the output of the trunk and outputs 64 dimensional embeddings
# embedder = torch.nn.DataParallel(MLP([trunk_output_size, 200]).to(device))
embedder = MLP([trunk_output_size, 200]).to(device)

# Set optimizers
trunk_optimizer = torch.optim.Adam(trunk.parameters(), lr=0.00001, weight_decay=0.0001)
embedder_optimizer = torch.optim.Adam(
    embedder.parameters(), lr=0.0001, weight_decay=0.0001
)

# Set the image transforms
train_transform = transforms.Compose(
    [
        # transforms.Resize(64),
        # transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=64),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        # transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

original_train = MyDataset("train")
original_val = MyDataset("val")
train_dataset = ClassDisjointMyDataset(original_train,original_val,True, train_transform)
val_dataset = ClassDisjointMyDataset(original_train,original_val,False, val_transform)

# Set the loss function
# loss = losses.TripletMarginLoss(margin=0.2)
loss = losses.ContrastiveLoss(pos_margin=0, neg_margin=0.5)

# Set the mining function
miner = miners.MultiSimilarityMiner(epsilon=0.1)

# Set the dataloader sampler
sampler = samplers.MPerClassSampler(
    train_dataset.targets, m=4, length_before_new_iter=len(train_dataset)
)

# Set other training parameters
batch_size = 16
num_epochs = 30
print(num_epochs)
# Package the above stuff into dictionaries.
models = {"trunk": trunk, "embedder": embedder}
optimizers = {
    "trunk_optimizer": trunk_optimizer,
    "embedder_optimizer": embedder_optimizer,
}
loss_funcs = {"metric_loss": loss}
mining_funcs = {"tuple_miner": miner}

record_keeper, _, _ = logging_presets.get_record_keeper(
    "example_logs", "example_tensorboard"
)
hooks = logging_presets.get_hook_container(record_keeper)
dataset_dict = {"val": val_dataset}
model_folder = "example_saved_models"

curr_time = datetime.datetime.now()
save_name=str(datetime.datetime.strftime(curr_time,'%Y-%m-%d_%H:%M:%S.json'))
save_name = R"MetricLossOnly_Contrastive_{}_{}".format(num_epochs,save_name)


save_file = os.path.join(os.path.abspath(os.getcwd()),save_name)
def end_of_testing_hook(tester):
    with open(save_file, 'a', encoding='utf-8') as f:
        # 将dic dumps json 格式进行写入
        f.write(json.dumps(tester.all_accuracies))
        f.write('\r\n')
    print(tester.all_accuracies)

def visualizer_hook(umapper, umap_embeddings, labels, split_name, keyname, *args):
    logging.info(
        "UMAP plot for the {} split and label set {}".format(split_name, keyname)
    )
    label_set = np.unique(labels)
    num_classes = len(label_set)
    fig = plt.figure(figsize=(20, 15))
    plt.gca().set_prop_cycle(
        cycler(
            "color", [plt.cm.nipy_spectral(i) for i in np.linspace(0, 0.9, num_classes)]
        )
    )
    for i in range(num_classes):
        idx = labels == label_set[i]
        plt.plot(umap_embeddings[idx, 0], umap_embeddings[idx, 1], ".", markersize=1)
    plt.show()


# Create the tester
tester = testers.GlobalEmbeddingSpaceTester(
    end_of_testing_hook=end_of_testing_hook,
    # visualizer=umap.UMAP(),
    # visualizer_hook=visualizer_hook,
    dataloader_num_workers=2,
    accuracy_calculator=AccuracyCalculator(k = 200),
)

end_of_epoch_hook = hooks.end_of_epoch_hook(
    tester, dataset_dict, model_folder, test_interval=1, patience=1
)



trainer = trainers.MetricLossOnly(
    models,
    optimizers,
    batch_size,
    loss_funcs,
    mining_funcs,
    train_dataset,
    sampler=sampler,
    dataloader_num_workers=4,
    end_of_iteration_hook=hooks.end_of_iteration_hook,
    end_of_epoch_hook=end_of_epoch_hook,
)

trainer.train(num_epochs=num_epochs)


# all_accuracies = tester.test(dataset_dict, num_epochs, trunk, embedder)

# curr_time = datetime.datetime.now()
# save_name=str(datetime.datetime.strftime(curr_time,'%Y-%m-%d_%H:%M:%S.json'))
# save_name = R"MetricLossOnly_Contrastive_{}_{}_{}".format(num_epochs,all_accuracies["val"]['precision_at_1_level0'],save_name)

# tf = open(save_name, "w")
# json.dump(all_accuracies,tf)
# tf.close()
# os.system("export $(cat /proc/1/environ |tr '\\0' '\\n' | grep MATCLOUD_CANCELTOKEN)&&/public/script/matncli node cancel -url https://matpool.com/api/public/node")
