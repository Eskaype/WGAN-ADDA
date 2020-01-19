from datasets.glaucoma_dataset import make_dataset
from torch.utils.data import DataLoader
import numpy as np

# split_path = {"origa": ['Origa_shuffled_train_images.txt' , 'Origa_shuffled_test_images.txt'],
#               "refuge": ['Refuge_shuffled_train_images.txt', 'Refuge_shuffled_test_images.txt'],
#               "drishti": ['Refuge_shuffled_val_images.txt', 'Refuge_shuffled_val_images.txt']}
def _init_fn(worker_id):
    np.random.seed(int(1+worker_id))

def make_weights_for_balanced_classes(images, nclasses):
    count = [0] * nclasses
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N/float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight

def make_data_loader(args, **kwargs):


    if args.dataset == 'glaucoma':
        train_set = make_dataset(args, split='train')
        val_set = make_dataset(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory= True, worker_init_fn=_init_fn, drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory= True, drop_last=True, worker_init_fn=_init_fn)
        test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory= False, drop_last=True)
        print(len(train_loader), len(val_loader), len(test_loader))
        return train_loader, val_loader, test_loader, num_class
    elif args.dataset == ['refuge', 'origa', 'drishti']:
        train_set = make_dataset(args, split='train', [ROOT_PATHS[args.dataset[0],
                                                        ROOT_PATHS[args.dataset[1]]])
    else:
        raise NotImplementedError
