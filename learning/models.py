from header import *

def set_features_extraction(model, ext_feat):
    if(ext_feat):
        for param in model.parameters():
            param.requires_grad = False

def get_other_model(model_name, num_classes):

    model_ft = None
    use_pretrained = True
          
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif(model_name=="resnet50"):
        model_ft = models.resnet50(pretrained=use_pretrained)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg11_bn":
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif(model_name=="vgg19"):
        model_ft = models.vgg19(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier[3].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet1_0":
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
    
    elif(model_name=="squeezenet1_0b"):
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        
        model_ft.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        model_ft.forward = lambda x: model_ft.classifier(model_ft.features(x)).view(x.size(0), num_classes)

    elif model_name == "densenet":
        model_ft = models.densenet121(pretrained=use_pretrained)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "inception":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)

    else:
        print("Invalid model name, exiting...")
        exit()

    set_features_extraction(model_ft, ext_feat)

    return model_ft

class custnet01(nn.Module):
    def __init__(self,num_classes):
        super(custnet01, self).__init__()
        self.dpout1 = 0.1
        self.nneur1 = 100
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 29 * 29, 10)
        self.fc2 = nn.Sequential(
            nn.Linear(10,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc4 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc5 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc6 = nn.Linear(self.nneur1,num_classes)

    def forward(self, x):
        x = self.pool(nnFunc.relu(self.conv1(x)))
        x = self.pool(nnFunc.relu(self.conv2(x)))
        x = x.view(-1, 16 * 29 * 29)
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = nnFunc.relu(self.fc3(x))
        x = nnFunc.relu(self.fc4(x))
        x = nnFunc.relu(self.fc5(x))
        x = nnFunc.relu(self.fc6(x))
        return x

class mernet01(nn.Module):
    
    def __init__(self,num_classes):
        self.nneur1 = 1000
        self.nfeat1 = 48
        self.dpout1 = 0.5
        self.num_classes = num_classes

        super(mernet01, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(self.nfeat1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc4 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc5 = nn.Linear(self.nneur1,self.num_classes)

    def forward(self, x):
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = nnFunc.relu(self.fc3(x))
        x = nnFunc.relu(self.fc4(x))
        x = self.fc5(x)
        x = nnFunc.log_softmax(x)
        return x

class mernet02(nn.Module):
    
    def __init__(self,num_classes):
        self.nneur1 = 1000
        self.nfeat1 = 48
        self.dpout1 = 0.1
        self.num_classes = num_classes

        super(mernet02, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(self.nfeat1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc4 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc5 = nn.Linear(self.nneur1,self.num_classes)

    def forward(self, x):
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = nnFunc.relu(self.fc3(x))
        x = nnFunc.relu(self.fc4(x))
        x = self.fc5(x)
        x = nnFunc.log_softmax(x)
        return x

class resnet50ext(nn.Module):
    def __init__(self, num_classes):
        super(resnet50ext,self).__init__()
        self.dpout1 = 0.5
        self.nneur1 = 1000
        self.model_ft = models.resnet50(pretrained=True)
        set_features_extraction(self.model_ft,False)
        self.num_ftrs = self.model_ft.fc.in_features
        modules = list(self.model_ft.children())[:-1]
        self.father = nn.Sequential(*modules)

        self.fc1 = nn.Sequential(
            nn.Linear(self.num_ftrs,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc3 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc4 = nn.Sequential(
            nn.Linear(self.nneur1,self.nneur1),
            nn.BatchNorm1d(num_features=self.nneur1),
            nn.Dropout(self.dpout1)
            )
        self.fc5 = nn.Linear(self.nneur1,num_classes)

    def forward(self, x):
        x = nnFunc.relu(self.father(x))
        x = x.view(-1, self.num_ftrs)
        x = nnFunc.relu(self.fc1(x))
        x = nnFunc.relu(self.fc2(x))
        x = nnFunc.relu(self.fc3(x))
        x = nnFunc.relu(self.fc4(x))
        x = nnFunc.relu(self.fc5(x))
        return x
        
def choose_model(model_name,num_classes,use_gpu):

    model_class = getattr(sys.modules[__name__], model_name)
    model = model_class(num_classes)
    
    if(use_gpu):
        model.cuda()
        
    return model