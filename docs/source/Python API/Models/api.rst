.. DEEG documentation master file, created by
   sphinx-quickstart on Thu Jul 22 11:23:59 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Models 
================================

.. py:function:: deeg.models.ConvNet(in_channels, num_classes)

   Use CNN classifier to classify original EEG signal
   Structure of CNN：3 layers of CNN and 2 layers of FC
   Loss and optimizer:
   criterion = nn.CrossEntropyLoss()
   optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
   scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.95)

   :param in_channels: the number of channels of 1st CNN layer, corresponding to the number of electrons
   :type in_channel: int
   :param num_classes: the number of classes seted for classification
   :type num_classes: int
   :return: a class of nn.Module, with defined CNN structure
   