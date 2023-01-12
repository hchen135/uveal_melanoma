import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import json
import torch
import numpy as np
import time,datetime
import pandas as pd
from sklearn.metrics import classification_report,roc_auc_score

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    EnsureType,
    Transpose
)
from monai.utils import set_determinism

from dataset import CellDataset
from util import time_estimation,time_convertion
from model import MLP

DATA_DIR = "C:\\Project\\Cell_method\\Data"
root_dir = ".\\models"
num_class=2
learning_rate = 1e-3
max_epochs = 200
val_interval = 2
batch_size=8
num_workers=1
input_dimension = 256
max_cells = 500



## Transformations
train_transforms = Compose(
    [
        LoadImage(image_only=True),
        # AddChannel(),
        # Printt(),
        Transpose((2,0,1)),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
        EnsureType(),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), 
    Transpose((2,0,1)),
     # AddChannel(), 
     ScaleIntensity(), 
     EnsureType()])

y_pred_trans = Compose([EnsureType(), Activations(softmax=True)])
y_trans = Compose([EnsureType(), AsDiscrete(to_onehot=num_class)])


def ModelTrain(train_ds,train_loader,epoch,max_epoch,time_start):
    print("start training")
    model.train()
    epoch_loss = 0
    step = 0
    print("epoch:", epoch)
    while(step < len(train_ds) / batch_size + 1):
        batch_data = train_ds.out(phase="train")
        step += 1
        inputs, labels = batch_data[0], batch_data[1]
        # print("inputs:",inputs)
        print(len(inputs))
        print(len(inputs[0]))
        # print("labels:",labels)
        inputs = [torch.tensor(i).to(device) for i in inputs]
        labels = torch.tensor(labels).to(device).reshape(-1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        epoch_len = len(train_ds) // train_loader.batch_size
        cur_time = time.time()
        time_passed,time_future = time_estimation(time_start,cur_time,epoch,max_epoch,step,epoch_len)

        print(
            f"{step}/{len(train_ds) // train_loader.batch_size}, "
            f"train_loss: {loss.item():.4f}, "
            f"time passed: {time_passed[0]}h{time_passed[1]}m{time_passed[2]}s, "
            f"time remain: {time_future[0]}h{time_future[1]}m{time_future[2]}s, ")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")



def ModelEval(val_ds,val_loader,phase = "val"):
    global best_metric
    global best_metric_epoch
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.eval()

    with torch.no_grad():

        y_pred = torch.tensor([], dtype=torch.float32, device=device)
        y = torch.tensor([], dtype=torch.long, device=device)
        val_names = []
        step = 0
        for i in range(len(val_ds)):
            val_data = val_ds.out(phase="test",index=i)
            step+=1
            val_features, val_labels, val_name = (
                val_data[0],
                val_data[1],
                val_data[2]
            )
            val_features = [torch.tensor(i).to(device) for i in val_features]
            val_labels = torch.tensor(val_labels).to(device).reshape(-1)
            y_pred = torch.cat([y_pred, (torch.nn.Softmax(dim=1)(model(val_features)))[0,1].reshape(-1)], dim=0)
            y = torch.cat([y, val_labels], dim=0)
            for name in val_name:
                val_names.append(name)
            print(
            f"{step}/{len(val_ds)}")
            # print(val_names[0].split('\\')[-2])

        print(y_pred)
        print(y)

        y_pred = y_pred.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

        result = roc_auc_score(y,y_pred)
        metric_values.append(result)
        acc_value = (y_pred >= 0.5  - y) == 0
        acc_metric = np.sum(acc_value) / len(acc_value)
        if phase == "val":
            if result > best_metric:
                best_metric = result
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(
                    root_dir, "best_metric_model.pth"))
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current slide_level AUC: {result:.4f}"
                f" current tile accuracy: {acc_metric:.4f}"
                f" best slide-level AUC: {best_metric:.4f}"
                f" at epoch: {best_metric_epoch}"
            )
        elif phase == "test":
            # TODO change to numpy array
            print(
                f"Best slide_level AUC: {result:.4f}"
            )
            y_pred = y_pred.astype(np.float64)
            y = y.astype(int)
            out_dict = {}
            out_array = []
            for i in range(len(val_ds)):
                out_dict[val_names[i]] = {"final_prediction":y_pred[i],"GT_class":int(y[i])}
                out_array.append([val_names[i],y_pred[i],int(y[i])])


            with open("result/result.json","w") as a:
                json.dump(out_dict,a,indent=4)
            pd.DataFrame(out_array,columns=["slide","prediction","Ground Truth"]).to_csv("result/result.csv",index=False)




if __name__ == '__main__': 
    print_config()

    ## Create dataset
    with open("split/train_subject.json") as a:
        train_subject_dict = json.load(a)
    with open("split/test_subject.json") as a:
        test_subject_dict = json.load(a)


    train_ds = CellDataset(train_subject_dict, DATA_DIR,batch_size,max_cells)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_ds = CellDataset(test_subject_dict, DATA_DIR,batch_size,max_cells)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers)


    ## Construct the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("#"*20)
    print(device)
    print("#"*20)
    #model = DenseNet121(spatial_dims=2, in_channels=3,
    #                    out_channels=num_class).to(device)
    model = MLP(input_dimension,device)
    model.to(device)
    print("model built")

    # loss_function = torch.nn.CrossEntropyLoss(weight=torch.Tensor([1.5,0.5]).float().to(device))
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    auc_metric = ROCAUCMetric()

    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    time_start = time.time()

    for epoch in range(max_epochs):

        ModelTrain(train_ds,train_loader,epoch,max_epochs,time_start)

        if (epoch + 1) % val_interval == 0:
            ModelEval(val_ds,val_loader,phase="val")
        print(epoch)

    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")

    ## test
    model.load_state_dict(torch.load(
        os.path.join(root_dir, "best_metric_model.pth")))
    ModelEval(val_ds,val_loader,phase="test")
