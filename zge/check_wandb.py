import os
import wandb
import pandas as pd
from datetime import datetime

api = wandb.Api()

# # Find the run folder and run file
# wandb_file = os.path.join(os.getcwd(), "experiments", "tensorboard_logs", "SPMamba-Echo2Mix", "wandb/latest-run/run-rce0uz19.wandb")
# assert os.path.isfile(wandb_file), f"wandb file: {wandb_file} does not exist!"

# set wandb dir
wandb_dir = os.path.join(os.getcwd(), "experiments", "tensorboard_logs", "SPMamba-Echo2Mix", "wandb")

# get run dir and run id
run_dir = os.path.join(wandb_dir, "latest-run")
run_dir = os.path.realpath(run_dir)
assert os.path.isdir(run_dir), f"run dir: {run_dir} does not exist!"
run_id = os.path.basename(run_dir).split('-')[-1] # rce0uz19, qard84oh
print(f'run id: {run_id}')

# # override run dir and run id
# run_id = 'rce0uz19'
# matched_foldernames = [d for d in os.listdir(wandb_dir) if os.path.isdir(os.path.join(wandb_dir, d)) and run_id in d]
# assert len(matched_foldernames) == 1, 'no or multiple matched dir(s)!'
# run_dir = os.path.join(wandb_dir, matched_foldernames[0])

# df = pd.read_json(wandb_file, lines=True)
username = "gezhenhaonuaa-sri-international"
project = "Real-work-dataset"
run = api.run(os.path.join(username, project, 'runs', run_id))
# run = api.run().from_path(wandb_file)

# history = run.history(keys=["epoch", "train_loss_step", "val_loss/dataloader_idx_0", "lr", "train_loss_epoch"])
# df = pd.DataFrame(history)

history_train_loss = run.history(keys=["epoch", "train_loss_epoch"])
train_loss_dict = dict(zip(history_train_loss["epoch"], history_train_loss["train_loss_epoch"]))

history_val_loss = run.history(keys=["epoch", "val_loss/dataloader_idx_0"])
val_loss_dict = dict(zip(history_val_loss["epoch"], history_val_loss["val_loss/dataloader_idx_0"]))

df_train_loss = pd.DataFrame(list(train_loss_dict.items()), columns=["Epoch", "Train Loss"])
df_val_loss = pd.DataFrame(list(val_loss_dict.items()), columns=["Epoch", "Val Loss"])
df_combined = pd.merge(df_train_loss, df_val_loss, on="Epoch")

# show the min train and val losses
train_loss_min = df_combined['Train Loss'].min()
val_loss_min = df_combined['Val Loss'].min()
print(f'min. train loss: {train_loss_min:.2f}')
print(f'min. val loss: {val_loss_min:.2f}')

# set result dir
result_dir = os.path.join(run_dir, "results")
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)
    print(f'created result dir: {result_dir}')
else:
    print(f'used existing result dir: {result_dir}')

# save tran/val loss from data frame to csv      
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
csvfile = os.path.join(result_dir, f"loss_per_epoch_{timestamp}.csv")
df_combined.to_csv(csvfile, index=False)

