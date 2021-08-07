Deep learning with DEEG
=======================

| In this tutorial, we’ll introduce how to use DEEG package to conduct
  deep learning (using PyTorch)
| Here we use basic 1d CNN model as an example.
| 1. Data Loading
| 2. Quality check of data 3. Model building
| 4. Load data with PyTorch 5. Model training and evaluation

**Step 1:** Load deeg and pytorch package, together necessary
dependencies

.. code:: ipython3

    # !pip install deeg
    import os
    import deeg
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader

| **Step 2:** Load dataset. Here we used commonly-used DEAP dataset as
  an example.
| After loading, the program will outout the shape of data and labels

.. code:: ipython3

    # Specify the directory of your downloaded DEAP dataset below
    data_dir = "/data1/ljq/datasets/EEG/DEAP/"
    deap_dataset = deeg.load_DEAP(data_dir)
    deap_data, deap_label = deap_dataset[0], deap_dataset[1].astype("int")


.. parsed-literal::

    Shape of data: (1240, 40, 8064)
    Shape of labels: (1240, 4)


| **Step 3:** Check whether the loaded dataset has missing values.
| If missing value (NaN) occurs (“check_result” is not none), you may
  need to check your data.

.. code:: ipython3

    check_result = deeg.process.data_quality_check(deap_data)
    print(check_result=={})


.. parsed-literal::

    *** Data Quality Check ***
    True


| **Step 4:** Build deep learning model.
| Here we use a built-in CNN model. The CNN model receives the signal
  from 40 channels simultaneously, and does convolutional calculation on
  1d array. The output of CNN model is a vector, corresponding to the
  probability of each category.
| Notice that we show GPU version here. You can also use CPU only by
  disable the “.cuda( )” command.

First, let’s set configurations for deep learning.

.. code:: ipython3

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size = 8
    lr = 0.001
    epochs = 100

| Then, build the model and set arguments corresponding to DEAP dataset
  (i.e. 40 channels and 9 output categories)
| And put the model to GPU device.

.. code:: ipython3

    model = deeg.models.ConvNet(in_channel=40, num_classes=9)
    model = model.cuda()

| **Step 5:** Load data into PyTorch loaders
| Here we just follow the workflow of PyTorch, using *Dataset* and
  *DataLoader*.

For illustration convenience, we use *sklearn* to split training and
validation set. We strongly recommend that you complete this procedure
by yourself.

.. code:: ipython3

    class DeapDataset(Dataset):
        def __init__(self, data_mat, label_mat, transform=None):
            self.data_mat = data_mat
            self.label_mat = label_mat
            self.transform = transform
        def __getitem__(self, idx):
            data = self.data_mat[idx,:]
            label = self.label_mat[idx,...]   # label_mat can also be of dimension 1
            return torch.tensor(data, dtype=torch.float), torch.tensor(label, dtype=torch.long)
        def __len__(self):
            return len(self.label_mat)

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    
    deap_label_one = deap_label[:,0]   # DEAP has 4 categories of emotions, we focus on the first
    train_data, val_data, train_label, val_label = train_test_split(deap_data, deap_label_one-1, 
                                                                    test_size=0.33, random_state=0)
    
    train_dataset = DeapDataset(train_data, train_label)
    val_dataset = DeapDataset(val_data, val_label)
    train_loader = DataLoader(train_dataset, batch_size=8, num_workers=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=8, num_workers=4, shuffle=False)

.. code:: ipython3

    class ConvNet(nn.Module):
        def __init__(self, in_channel=40, num_classes=1):
            super(ConvNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv1d(in_channel, 32, kernel_size=5, stride=(1), padding=(2), bias=True),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, stride=(1), padding=(2), bias=True),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=(1), padding=(1), bias=True),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(),
                nn.AvgPool1d(kernel_size=2, stride=2)
            )
            self.fc1 = nn.Linear(128 * 1008, 256, bias=True)
            self.fc2 = nn.Linear(256, num_classes, bias=True)
    
        def forward(self, x):
            out = self.layer1(x)
            out = self.layer2(out)
            out = self.layer3(out)
            out = out.reshape(out.size(0), -1)
            out = self.fc1(out)
            out = self.fc2(out)
            return out

**Step 6:** Set the loss function and optimizer

.. code:: ipython3

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

**Step 7:** Model training and evaluation.

.. code:: ipython3

    def train(epoch, train_loader):
        model.train()
        train_loss = 0
        for data, label in train_loader:
            data, label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.detach()*data.size(0)
        train_loss = train_loss/len(train_loader.dataset)
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
    
    def val(epoch, val_loader):                          
        model.eval()
        val_loss = 0
        acc = 0
        with torch.no_grad():
            for data, label in val_loader:
                data, label = data.cuda(), label.cuda()
                out = model(data)
                loss = criterion(out, label)
                val_loss += loss.detach()*data.size(0)
                preds = out.argmax(dim=1)
                acc += torch.sum(preds==label)
    #             print(metric_acc(out_mask, mask), metric_acc(preds, label))
        val_loss = val_loss/len(val_loader.dataset)
        acc = acc/len(val_loader.dataset)
        print('Epoch: {} \tValidation Loss: {:.6f},  ACC: {:.6f}'.format(epoch, val_loss, acc))


.. code:: ipython3

    for epoch in range(1,epochs+1):
        train(epoch, train_loader)
        val(epoch, val_loader)
