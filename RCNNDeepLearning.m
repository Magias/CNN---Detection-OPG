%imds = imageDatastore('C:\CNN-OPG\zuby','LabelSource','foldernames');

data = load('imds.mat');
teethDataset = data.imds.Files;

teethDataset(:)
