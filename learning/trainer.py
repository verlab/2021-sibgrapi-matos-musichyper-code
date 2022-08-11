from header import *

def get_class_amounts(group_list,num_classes):
    class_amounts = [0]*num_classes
    for item in group_list:
        class_amounts[item[1]]+=1
    return class_amounts

def balance_group_list(group_list,num_classes):
    random.shuffle(group_list)
    class_amounts = [0]*num_classes
    class_amounts_ceif = [0]*num_classes
    for item in group_list:
        class_amounts[item[1]]+=1
    min_amt = min(class_amounts)
    group_list_bal = []
    for item in group_list:
        if(class_amounts_ceif[item[1]]<min_amt):
            class_amounts_ceif[item[1]]+=1
            group_list_bal.append(item)
    return group_list_bal

def split_individual_list(in_list,per1,per2,per3):
    per1,per2,per3 = per1/100,per2/100,per3/100
    list1,list2,list3 = [],[],[]
    at = len(in_list)
    a1,a2,a3 = int(per1*at),int(per2*at),int(per3*at)
    random.shuffle(in_list)
    list1 = in_list[0:a1]
    list2 = in_list[a1:a1+a2]
    list3 = in_list[a1+a2:a1+a2+a3]
    return (list1,list2,list3)

def split_dataset_generic(dataset_dir,split_dir,tot_subper,per1,per2,per3,undsamp,labsuf=""):

    filename = dataset_dir+"labels"+labsuf+".csv"

    print("Generating splits...")

    data_table = pd.read_csv(filename)

    names = data_table.iloc[:,0]
    classes = data_table.iloc[:,1]

    num_classes = max(classes)+1
    print("Detected num classes: "+str(num_classes))

    full_list = []
    for i in range(len(names)):
        utils.show_progress("Reading items... ",i,len(names))
        full_list.append((dataset_dir+names.iloc[i],classes.iloc[i]))

    print("Total samples..: "+str(get_class_amounts(full_list,num_classes)))

    full_list = full_list[:int((tot_subper/100)*len(full_list))]

    print("Undsmp samples..: "+str(get_class_amounts(full_list,num_classes)))

    if(undsamp==True):
        full_list = balance_group_list(full_list,num_classes)

    per_class_lists = []
    for i in range(num_classes):
        per_class_lists.append([])

    for i in range(len(full_list)):
        filename = full_list[i][0]
        label = full_list[i][1]
        per_class_lists[label].append((filename,label))

    train_list = []
    valid_list = []
    test_list = []
    
    for pc_list in per_class_lists:
        list1,list2,list3 = split_individual_list(pc_list,per1,per2,per3)
        for item in list1:
            train_list.append(item)
        for item in list2:
            valid_list.append(item)
        for item in list3:
            test_list.append(item)

    print("Train samples..: "+str(get_class_amounts(train_list,num_classes)))
    print("Valid samples..: "+str(get_class_amounts(valid_list,num_classes)))
    print("Test samples...: "+str(get_class_amounts(test_list ,num_classes)))

    random.shuffle(train_list)
    random.shuffle(valid_list)
    random.shuffle(test_list)

    print("Saving splits...")

    (pd.DataFrame(full_list )).to_csv(split_dir+"full_list.csv" ,index=None,header=False)
    (pd.DataFrame(train_list)).to_csv(split_dir+"train_list.csv",index=None,header=False)
    (pd.DataFrame(valid_list)).to_csv(split_dir+"valid_list.csv",index=None,header=False)
    (pd.DataFrame(test_list )).to_csv(split_dir+"test_list.csv" ,index=None,header=False)

def calc_loss_weights(sample_list,num_classes):

    sample_amount = [0]*num_classes
    loss_weights = [0]*num_classes
    total_samples = 0
    
    for sample in sample_list:
        label = int(sample[1])
        sample_amount[label]+=1
        total_samples+=1

    for i in range(len(sample_amount)):
        if(sample_amount[i]==0):
            loss_weights[i] = 0
        else:
            loss_weights[i] = total_samples/sample_amount[i]

    loss_weights = [float(i)/sum(loss_weights) for i in loss_weights]
    loss_weights_cut = []
    for v in loss_weights:
        loss_weights_cut.append(float(f"{v:.4f}"))

    return loss_weights

def loader_to_cuda(group_loader,group_name,use_gpu):
    if(use_gpu):
        group_loader_out = []
        for i, (inputs,labels) in enumerate(group_loader):
            utils.show_progress("  Preload "+str(group_name)+" Batches... ",i,len(group_loader))
            inputs = inputs.cuda()
            labels = labels.cuda()
            group_loader_out.append((inputs,labels))
    else:
        group_loader_out = group_loader
    return group_loader_out

def batch_to_cuda(inputs,labels,use_gpu):
    if(use_gpu):
        inputs = inputs.cuda()
        labels = labels.cuda()
    return inputs,labels

def load_dataset_group(group_pathlist,num_classes):

    data_list = []
    data_table = pd.read_csv(group_pathlist)

    labels_amount = [0]*num_classes
    for i in range(data_table.shape[0]):
        image_path = data_table.iloc[i,0]
        image_label_c = data_table.iloc[i,1]
        labels_amount[image_label_c]+=1
        data_list.append((image_path,image_label_c))

    return data_list

def train_model(model,optimizer,train_loader,valid_loader,criterion,model_statedict,
optim_statedict,max_num_epochs,early_stop_lim,save_interval,preload_dataset,log_filename,use_gpu):

    print("Preparing to train...")

    min_valid_loss = 100
    valid_loss = min_valid_loss

    if(preload_dataset==True):
        train_loader = loader_to_cuda(train_loader,"Train",use_gpu)
        valid_loader = loader_to_cuda(valid_loader,"Valid",use_gpu)

    early_stop_i=0

    for epoch in range(max_num_epochs):
        #Train step:
        model.train()
        utils.save_log("Epoch {}:".format(epoch+1),log_filename)
        correct,total = 0,0
        for i, (inputs, labels) in enumerate(train_loader):
            utils.show_progress("  Training... ",i,len(train_loader))
            if(preload_dataset==False):
                inputs,labels = batch_to_cuda(inputs,labels,use_gpu)
            outputs = model(inputs)
            train_loss = criterion(outputs, labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        train_acc = int(100*correct/total)

        #Evaluation step:
        model.eval()
        with torch.no_grad():
            correct,total = 0,0
            for i, (inputs, labels) in enumerate(valid_loader):
                utils.show_progress("  Validating... ",i,len(valid_loader))
                if(preload_dataset==False):
                    inputs,labels = batch_to_cuda(inputs,labels,use_gpu)
                outputs = model(inputs)
                valid_loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            valid_acc = int(100*correct/total)

            log_str = ("    T-Loss: {:.4f}, V-Loss: {:.4f}, T-Acc: {}, V-Acc: {}"
            .format(train_loss.item(),valid_loss.item(),train_acc,valid_acc))

            utils.save_log(log_str,log_filename)
        
        if(((epoch+1)%save_interval==0 and valid_loss<min_valid_loss) or early_stop_lim==-1):
            utils.save_log("      Saving... " +str(model_statedict)+","+str(optim_statedict),log_filename)
            torch.save(model.state_dict(), model_statedict)
            torch.save(optimizer.state_dict(), optim_statedict)
            min_valid_loss = valid_loss
            early_stop_i=0

        early_stop_i+=1
        if(early_stop_i>early_stop_lim and early_stop_lim!=-1):
            utils.save_log(" Stopped due not evolving!",log_filename)
            break

        if(epoch>=max_num_epochs-1):
            utils.save_log("Max num epochs achieved!",log_filename)
    

def test_model(model,model_statedict,test_loader,num_classes,preload_dataset,model_dir,use_gpu):

    grndtt_full = []
    predic_full = []

    model.load_state_dict(torch.load(model_statedict,map_location=torch.device('cuda' if use_gpu else 'cpu')))

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (inputs, labels) in enumerate(test_loader):
            utils.show_progress("  Testing... ",i,len(test_loader))
            inputs,labels = batch_to_cuda(inputs,labels,use_gpu)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            grndtt_full += labels.tolist()
            predic_full += predicted.tolist()
            correct += (predicted == labels).sum().item()

        test_accuracy = 100*correct/total
        print ("Test Accuracy: {}".format(test_accuracy))

    confusion_mat = np.zeros((num_classes, num_classes), dtype=int)
    np.add.at(confusion_mat, [grndtt_full, predic_full], 1)
    
    target_names = []
    for i in range(num_classes):
        target_names.append(str(i))

    print(confusion_mat)

def predict_list(model,model_statedict,input_list,data_type,use_gpu):
    
    predic_list = []
    for item in input_list:
        predic_list.append((item,-1))
    data_loader = learning.loader.c
    predic_dataset = data_loader(predic_list,data_type)
    predic_loader = torch.utils.data.DataLoader(dataset=predic_dataset, 
    batch_size=10, num_workers = num_cores, shuffle=False)
    output_list = []
    
    model.load_state_dict(torch.load(model_statedict,map_location=torch.device('cuda' if use_gpu else 'cpu')))

    model.eval()
    output_list = []
    with torch.no_grad():
        for i, (inputs,labels) in enumerate(predic_loader):
            utils.show_progress("  Predicting "+data_type+" labels... ",i,len(predic_loader))
            inputs,labels = batch_to_cuda(inputs,labels,use_gpu)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            output_list += predicted.tolist()

    return output_list

def run(task,config_version,use_gpu):

    model_dir = models_dir+str(config_version)+"/"
    if not(os.path.exists(model_dir)):
        os.makedirs(model_dir)

    if(task=='clean'):
        option = str(input("Delete model dir for "+str(config_version)+" version? (y/n) "))
        if(option=='y'):
            shutil.rmtree(model_dir)
        return

    config_file = model_dir+"config.json"
    if(os.path.exists(config_file)):
        config_dict = json.load(open(config_file))
    else:
        config_dict = default_config_dict
        json.dump(config_dict,open(config_file,'w'))

    if(manual_config):
        config_dict["dataset_name"] = input(" dataset_name: ")
        config_dict["model_name"] = input("  model_name: ")
        config_dict["batch_size"] = int(input("  batch_size: "))
        config_dict["labels_suffix"] = input(" labels_suffix: ")
        config_dict["preload_dataset"] = utils.str2bool(input("preload_dataset: "))
        json.dump(config_dict,open(config_file,'w'))

    print("Train configuration: ")
    print("  Model dir: "+str(model_dir))
    for item in config_dict:
        print("  "+item+": "+str(config_dict[item]))

    model_name = config_dict["model_name"]
    dataset_name = config_dict["dataset_name"]
    labels_suffix = config_dict["labels_suffix"]
    preload_dataset = bool(config_dict["preload_dataset"])
    total_subpercent = config_dict["total_subpercent"]
    train_percent = config_dict["train_percent"]
    valid_percent = config_dict["valid_percent"]
    test_percent = config_dict["test_percent"]
    undersampling = bool(config_dict["undersampling"])
    max_num_epochs = config_dict["max_num_epochs"]
    early_stop_lim = config_dict["early_stop_lim"]
    save_interval = config_dict["save_interval"]
    learning_rate = config_dict["learning_rate"]
    weight_decay = config_dict["weight_decay"]
    batch_size = config_dict["batch_size"]

    data_type = "audio" if dataset_name=="DEAM" else "video"

    if(data_type=="video"):
        dataset_folder = image_dataset_dir+dataset_name+"/"
    else:
        dataset_folder = audio_dataset_dir+dataset_name+"/"
    
    model_statedict = model_dir+"model.pth"
    optim_statedict = model_dir+"optim.pth"
    log_filename = model_dir+"log.txt"

    print("Loading dataset...")

    labels_suffix = "_"+labels_suffix[0]

    if not(os.path.exists(model_dir+"full_list.csv")):
        split_dataset_generic(dataset_folder,model_dir,total_subpercent,
        train_percent,valid_percent,test_percent,undersampling,labels_suffix)

    data_table = pd.read_csv(model_dir+"full_list.csv")
    num_classes = max(data_table.iloc[:,1])+1

    model = learning.models.choose_model(model_name,num_classes,use_gpu)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate,weight_decay=weight_decay)

    train_list = load_dataset_group(model_dir+"train_list.csv",num_classes)
    valid_list = load_dataset_group(model_dir+"valid_list.csv",num_classes)
    test_list  = load_dataset_group(model_dir+"test_list.csv" ,num_classes)

    loss_weights = calc_loss_weights(train_list,num_classes)
    loss_weights = torch.tensor(loss_weights)
    if(use_gpu):
        loss_weights = loss_weights.cuda()

    #print("Loss weights: "+str(loss_weights))

    if(data_type=="video"):
        criterion = nn.CrossEntropyLoss(weight=loss_weights)
    else:
        criterion = nn.NLLLoss(weight=loss_weights)
    
    train_loader = torch.utils.data.DataLoader(dataset=learning.loader.c(train_list,data_type),
    batch_size=batch_size, num_workers = num_cores, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=learning.loader.c(valid_list,data_type),
    batch_size=batch_size, num_workers = num_cores, shuffle=False)
    test_loader  = torch.utils.data.DataLoader(dataset=learning.loader.c(test_list,data_type),
    batch_size=batch_size, num_workers = num_cores, shuffle=False)

    if(task=="train"):
        if(os.path.exists(model_statedict)):
            model.load_state_dict(torch.load(model_statedict,map_location=torch.device('cuda' if use_gpu else 'cpu')))
            optimizer.load_state_dict(torch.load(optim_statedict,map_location=torch.device('cuda' if use_gpu else 'cpu')))    
            print("Model already exists, continuing it ...")
        train_model(model,optimizer,train_loader,valid_loader,criterion,model_statedict,
        optim_statedict,max_num_epochs,early_stop_lim,save_interval,preload_dataset,log_filename,use_gpu)

    elif(task=="test"):
        if(os.path.exists(model_statedict)):
            test_model(model,model_statedict,test_loader,num_classes,preload_dataset,model_dir,use_gpu)
        else:
            print("No model found to test!")
            exit()
