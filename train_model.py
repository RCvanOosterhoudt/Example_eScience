# ---------------------- Testing libraries ----------------------
import ipdb

# ---------------------- Normal libraries ----------------------
import getopt
from monai.losses import DiceLoss, GeneralizedDiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.data import (
    Dataset,
    DataLoader,
    decollate_batch
)
from monai.transforms import (
    Compose,
    AsDiscrete,
)

from datetime import datetime
import os
import pandas as pd
import sys
import torch
import yaml

# ---------------------- Personal Libraries ----------------------
from data_augmentation import CreateTrainTransform, CreateValTransform
from data_load import CreateDataset
from Loss import (
    BressanLoss,
    OosterhoudtLoss,
    CrossEntropyLoss,
    WeightedCrossEntropyLoss,
    MSELoss,
    MAELoss,
)
from Loss.wrapper import LossWrapper
from Evaluate import EvaluationPipeline
from Model import model_load as ModelLoad
from Utils import logging_utils, plt_utils

# ---------------------- Load configuration file ----------------------
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# ---------------------- System arguments ----------------------
opts, args = getopt.getopt(sys.argv[1:], "hl:e:c:", ["ifile=", "ofile="])

for opt, arg in opts:
    if opt == "-h":
        print("train_model.py -l <lossfunction> -e  <max epoch")
        sys.exit()
    elif opt in ["-l", 'loss']:
        config['loss_name'] = arg
        # config.loss_name = arg
    elif opt in ["-e", "-epochs"]:
        config['max_epochs'] = int(arg)
        # config.max_epochs = int(arg)
    elif opt in ["-c", "-computer"]:
        if arg == 'GPU':
            config['data_dir'] = "XXX"  # path to folder on GPU cluster.
            config['save_dir'] = "XXX"  # path to save folder on GPU cluser.

# ---------------------- Result placeholders ----------------------
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []

# ---------------------- Create folders ----------------------
run_dir = os.path.join(config['save_dir'], datetime.now().strftime("%Y-%m-%d_%H:%M"))

os.mkdir(run_dir)

validation_path_png = os.path.join(run_dir, 'validation_images_png')
os.mkdir(validation_path_png)

validation_path_nii = os.path.join(run_dir, 'validation_images_nii')
os.mkdir(validation_path_nii)

model_save_path = os.path.join(run_dir, 'model')
os.mkdir(model_save_path)

# ---------------------- Maintain Consistency ----------------------
# set_determinism(seed=0)
CUDA_LAUNCH_BLOCKING = 1

# ---------------------- Model Set-up ----------------------
# Check if output channels matches the loss functions.
if 'MSE' in config['loss_name'] or 'MAE' in config['loss_name']:
    if len(config['loss_name']) > 1:
        assert config['output_channels'] == 3, ("When the loss consist of either MAE or MSE the output channels must "
                                                "be 3. Change it in the config.yaml file.")

    elif len(config['loss_name']) == 1:
        assert config['output_channels'] == 1, ("When only using MAE or MSE the output channels must be 1. Change it "
                                                "in the config.yaml file.")
else:
    assert config['output_channels'] == 2, ("When not using MEA or MSE the output channels must be 2. Change it in the "
                                            "config.yaml file")


if config['model_name'] == 'monai_UNet':
    model = ModelLoad.monai_unet(output_channels=config['output_channels'], dropout=config['dropout'])
elif config['model_name'] == 'XXX_UNet':  # XXX is name of coworker.
    model = ModelLoad.XXX_unet()  # XXX is name of coworker.
else:
    raise ValueError('Model name is unknown.')

# Select correct GPU-device. In my case "cuda:0".
device = "cuda:0"
try:
    model.to(device)
except:
    raise ValueError('Current machine does not have CUDA installed!')

# ---------------------- Model Optimizer ----------------------
# --- Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), float(config['learning_rate']))

# ---------------------- Evaluation metric ----------------------
evaluation_pipeline = EvaluationPipeline()

# ---------------------- Make from the label and prediction a one-hot-encoding --------------------
post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
post_label = Compose([AsDiscrete(to_onehot=2)])

# ---------------------- Data Set-up ----------------------
create_ds = CreateDataset(
    frac_Tr=config['fraction_Tr'],
    frac_Ts=config['fraction_Ts'],
    frac_Va=config['fraction_Va'],
    clean_split=False
)
train_files, val_files, test_files, all_files = create_ds(data_dir=config['data_dir'])

# ---------------------- Transform Set-up ----------------------
train_transforms = CreateTrainTransform(patch_size=config['patch_size'])
val_transforms = CreateValTransform()

# ---------------------- Loss functions ----------------------
loss_function_list, weight_list = [], []
loss_dict = {
    'Bressan': (
        BressanLoss,
        {'dataset': all_files,
         'mask': True,
         'weight': config['Imbalance_amplifier']}
    ),

    'Oosterhoudt_v1': (
        OosterhoudtLoss,
        {'dataset': all_files,
         'mask': True,
         'weight': config['Imbalance_amplifier']}
    ),

    'Dice': (
        DiceLoss,
        {'include_background': False,
         'to_onehot_y': True,
         'softmax': True}
    ),

    'GeneralizedDice': (
        GeneralizedDiceLoss,
        {'include_background': False,
         'to_onehot_y': True,
         'softmax': True}
    ),

    'CrossEntropy': (
        CrossEntropyLoss,
        {'include_background': False,
         'to_onehot_y': True,
         'gamma': 0,
         'use_softmax': True,
         'weight': None}
    ),

    'WeightedCrossEntropy': (
        WeightedCrossEntropyLoss,
        {'include_background': True,
         'to_onehot_y': True,
         'gamma': 0,
         'use_softmax': True,
         'weight': [0, 1]}
    ),

    'Focal': (
        FocalLoss,
        {'include_background': False,
         'to_onehot_y': True,
         'alpha': config['Focal_alpha'],
         'use_softmax': True}
    ),

    'MSE': (
        MSELoss, {}
    ),

    'MAE': (
        MAELoss, {}
    )

}

assert len(config['loss_name']) == len(config['loss_weight']), \
    ("Number of losses and weights is different. Change 'loss_name' and/or 'loss_weight' in the "
     "'config.py' file.")

for loss in config['loss_name']:
    loss_function_list.append(loss_dict[loss][0](**(loss_dict[loss][1])))

# ----- Notes for the losses -----
# MaskedDiceLoss
# TODO: Could we change the settings to improve the results.
#
# GeneralizedDiceLoss
# TODO: check out the GeneralizedDiceLoss function as alternative to Mas
#
# TODO: the current GeneralizedDiceLoss calculates the class imbalance b and the background. It
#  does NOT take only the mask into account. if 'Bressan' in config.loss_name:
#
# Cross Entropy
#
# TODO: for all cross entropy based loss function a softmax is automatically
#  included during the loss. For now the loss in the standard pipeline, causing softmax to be
#  applied twice!
#
# WeightedCrossEntropy
#
# TODO: Add the weight for the class imbalance. TODO: We have two options for the next line.
#  Either we set the include_background to False, but then we need only one weight for W. Or we
#  set include_background to True then we need to feed the function two weights. I do not know
#  which option would be better. TODO: See MaskedCrossEntropy about the mask.
#
# FocalLoss
# TODO: See MaskedCrossEntropy about the mask.

# ---------------------- DataLoader ----------------------
train_ds = Dataset(data=train_files, transform=train_transforms)
train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, num_workers=10)

val_ds = Dataset(data=val_files, transform=val_transforms)
val_loader = DataLoader(val_ds, batch_size=1, num_workers=10)

# ---------------------- Logger ----------------------
log = logging_utils.logger(os.path.join(run_dir, 'logfile.log'))

log.log_message('--- Variables ---')

for variable in config:
    if variable == 'break':
        break
    log.log_variable(variable_name=variable, variable=config[variable])

if config['loss_name'] == 'Bressan':
    log.log_variable(variable_name='Bressan lambda', variable=config['Imbalance_amplifier'])
if config['loss_name'] == 'Oosterhoudt':
    log.log_variable(variable_name='Oosterhoudt lambda', variable=config['Imbalance_amplifier'])
if config['loss_name'] == 'Focal':
    log.log_variable(variable_name='Focal Alpha', variable=config['Focal_alpha'])

log.log_line()
log.log_message('--- Training set ---')
log.log_variable(variable_name='Achieved fraction training',
                 variable=round(len(train_files) / len(all_files), 4))
log.log_variable(variable_name='Training data set', variable=train_files)

log.log_line()
log.log_message('--- Validation set ---')
log.log_variable(variable_name='Achieved fraction validating',
                 variable=round(len(val_files) / len(all_files), 4))
log.log_variable(variable_name='Validation data set', variable=val_files)

log.log_line()
log.log_message('--- Testing set ---')
log.log_variable(variable_name='Achieved fraction testing',
                 variable=round(len(test_files) / len(all_files), 4))

log.log_line()
log.log_message('--- Description ---')
log.log_message('Note down what you would like to achieve with this model/settings')
log.log_line()

log.log_message('--- Model Performance ---')

# ---------------------- Run model ----------------------
output_training = pd.DataFrame()
early_stop_count = 0
for epoch in range(config['max_epochs']):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{config['max_epochs']}")

    # ----- Train the model -----
    model.train()
    epoch_loss, step = 0, 0
    epoch_loss_separate = torch.zeros(len(config['loss_name'])).to(device)
    for batch_data in train_loader:
        step += 1
        inputs, labels, masks, edm, gdm = (
            batch_data["image"].to(device),
            batch_data["label"].to(device),
            batch_data["mask"].to(device),
            batch_data["EDM"].to(device).astype(torch.float32),
            batch_data["GDM"].to(device).astype(torch.float32)
        )

        if config['distance_map'] == "EDM":
            label_distance = edm
        elif config['distance_map'] == "GDM":
            label_distance = gdm
        else:
            label_distance = None

        # Checking input and output size.
        B, M, X, Y = inputs.shape
        _, K, _, _ = labels.shape
        assert labels.shape == masks.shape == (B, K, X, Y)

        # Set gradients back to zero.
        optimizer.zero_grad()

        # Run model with inputs.
        pred_logits = model(inputs)

        # Check predicted output size.
        assert pred_logits.shape == (B, config['output_channels'], X, Y), (pred_logits.shape, B, K, X, Y)

        # Only apply this split when 3 output channels is selected. In the case of MAE and MSE!
        if config['output_channels'] == 3:
            # First two channels are for- and background one-hot-encoding for the segmentation.
            pred_distance = pred_logits[:, 2, :, :].unsqueeze(1)
            pred_logits = pred_logits[:, :2, :, :]
        elif config['output_channels'] == 2:
            # When output_channels is 2, we only generate the one-hot encoding for the segmentation and so the
            # distance map prediction is set to 0.
            pred_distance = []
        else:
            pred_distance = pred_logits

        # Implement the mask just as in the MaskedDiceLoss implementation of Monai.
        if config['mask_loss']:
            # If labels is multiplied with masks after the concatenation it results in 2 channels, while it should be 1.
            labels = torch.einsum('ijkl, ijkl -> ijkl', labels, masks)

            masks = torch.concat((masks, masks), 1)
            pred_logits = torch.einsum('ijkl, ijkl -> ijkl', pred_logits, masks)

        # Calculate the loss.
        losses = [weight * LossWrapper(
            loss_function,
            input=pred_logits,
            target=labels,
            input_distance=pred_distance,
            target_distance=label_distance
        ).astype(torch.float32)
                  for weight, loss_function in zip(config['loss_weight'], loss_function_list)]

        # Sum the losses.
        loss = torch.sum(torch.stack(losses))

        # Update the model.
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        # If the total loss is based on multiple separate loss functions, save them also individually.
        if len(config['loss_name']) > 1:
            for n, loss in enumerate(losses):
                epoch_loss_separate[n] += loss

    # Calculate the average loss.
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.2e}")

    # ----- Log the loss -----
    # Total loss
    log.log_message(f"Epoch {epoch + 1} average loss: {epoch_loss:.2e}")

    # Separate loss, if more than 1 loss function is defined.
    if len(config['loss_name']) > 1:
        for n, loss in enumerate(epoch_loss_separate):
            log.log_message(f"Epoch {epoch + 1} average {config['loss_name'][n]} loss: {loss / step:.2e}")

    # ----- Evaluate the model performance -----
    output_epoch = pd.DataFrame()
    if (epoch + 1) % config['val_interval'] == 0:
        model.eval()
        with torch.no_grad():
            for n, val_data in enumerate(val_loader):
                # TODO would including the mask in the validation make sense? My first thought says NO.
                val_inputs, val_labels, val_masks = (
                    val_data["image"].to(device),
                    val_data["label"].to(device),
                    val_data["mask"].to(device)
                )

                info = {"Epoch": epoch + 1, "Epoch Loss": epoch_loss, "Validation sample": n}

                # Scan over the full image with the set patch_size.
                roi_size = (config['patch_size'], config['patch_size'])
                sw_batch_size = 4
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model,
                                                       overlap=config['inference_overlap'])

                # Work in progress...
                if config['output_channels'] != 2:
                    val_outputs = val_outputs[:, :2, :, :]

                # Last activation function to go to probabilities.
                val_prob = torch.softmax(val_outputs, dim=1)[0]

                # ---- Make outputs discrete -----
                val_output_list = [post_pred(i) for i in decollate_batch(val_outputs)]
                val_label_list = [post_label(i) for i in decollate_batch(val_labels)]

                # ----- Evaluation metrics -----
                results = evaluation_pipeline(
                    output=val_output_list,
                    label_list=val_label_list,
                    probability=val_prob,
                    label_tensor=val_labels,
                    other_info=info
                )

                # Add evaluation metrics to pandas dataframe.
                output_epoch = pd.concat([output_epoch, results], ignore_index=True)

            # ----- Catch size errors in the pandas dataframe -----
            try:
                output_training = pd.concat([output_training, output_epoch], ignore_index=True)
            except Exception:
                raise Exception

            # ----- Extract the metric of interest -----
            metric = output_epoch[config['metric_watch']].mean().item()
            metric_values.append(metric)

            # ----- Test if model is best performing -----
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(),
                           os.path.join(model_save_path, "best_metric_model_" + str(epoch + 1) +
                                        ".pth"))

                early_stop_count = 0
            else:
                early_stop_count += config['val_interval']
                print(early_stop_count)

                # ----- Implement early stopping -----
                if early_stop_count >= config['early_stop']:
                    log.log_message(f"Model stopped due to early stopping at epoch {epoch + 1}.")
                    break

            save_path = os.path.join(run_dir, 'Loss_curve.png')
            plt_utils.loss_and_metric(epoch_loss_values,
                                      metric_values,
                                      config['val_interval'],
                                      save_path)

            print(
                f"current epoch: {epoch + 1} current metric score: {metric:.4e}"
                f"\nbest metric score: {best_metric:.4e} "
                f"at epoch: {best_metric_epoch}"
            )

            log.log_message('_' * 10)
            log.log_message(f"--- Current epoch: {epoch + 1} current metric score: {metric:.4e}")
            log.log_message('_' * 10)

            output_training.to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)

print(f"train completed, best_metric: {best_metric:.4e} " f"at epoch: {best_metric_epoch}")

log.log_message(f"Training completed, best_metric: {best_metric:.4e} "
                f"at epoch: {best_metric_epoch}")

log.log_line()
