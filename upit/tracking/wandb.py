# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06_tracking.wandb.ipynb.

# %% auto 0
__all__ = ['SaveModelAtEndCallback', 'log_dataset', 'UPITWandbCallback']

# %% ../../nbs/06_tracking.wandb.ipynb 2
import wandb
from fastai.vision.all import *
from fastai.callback.wandb import *
from fastai.callback.wandb import _format_metadata, _format_config
from fastai.basics import *
from fastai.vision.gan import *
from ..models.cyclegan import *
from ..data.unpaired import *
from ..train.cyclegan import *
from ..metrics import *

# %% ../../nbs/06_tracking.wandb.ipynb 3
class SaveModelAtEndCallback(Callback):
    def __init__(self, fname='model', with_opt=False): store_attr()
    def _save(self, name): self.last_saved_path = self.learn.save(name, with_opt=self.with_opt)
    def after_fit(self, **kwargs): self._save(f'{self.fname}')
    @property
    def name(self): return "save_model"

# %% ../../nbs/06_tracking.wandb.ipynb 4
def log_dataset(main_path, folder_names=None, name=None, metadata={}, description='raw dataset'):
    "Log dataset folder"
    # Check if wandb.init has been called in case datasets are logged manually
    if wandb.run is None:
        raise ValueError('You must call wandb.init() before log_dataset()')
    path = Path(main_path)
    if not path.is_dir():
        raise f'path must be a valid directory: {path}'
    name = ifnone(name, path.name)
    _format_metadata(metadata)
    artifact_dataset = wandb.Artifact(name=name, type='dataset', metadata=metadata, description=description)
    # log everything in folder_names
    if not folder_names: folder_names = [p.name for p in path.ls() if p.is_dir()]
    for p in path.ls():
        if p.is_dir():
            if p.name in folder_names and p.name != 'models': artifact_dataset.add_dir(str(p.resolve()), name=p.name)
        else: artifact_dataset.add_file(str(p.resolve()))
    wandb.run.use_artifact(artifact_dataset)

# %% ../../nbs/06_tracking.wandb.ipynb 5
class UPITWandbCallback(Callback):
    "Saves model topology, losses & metrics"
    remove_on_fetch,order = True,Recorder.order+1
    # Record if watch has been called previously (even in another instance)
    _wandb_watch_called = False

    def __init__(self, log="gradients", log_preds=True, log_model=True, log_dataset=False, folder_names=None, dataset_name=None, valid_dl=None, n_preds=36, seed=12345, reorder=True):
        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError('You must call wandb.init() before WandbCallback()')
        # W&B log step
        self._wandb_step = wandb.run.step - 1  # -1 except if the run has previously logged data (incremented at each batch)
        self._wandb_epoch = 0 if not(wandb.run.step) else math.ceil(wandb.run.summary['epoch']) # continue to next epoch
        store_attr()

    def before_fit(self):
        "Call watch method to log model topology, gradients & weights"
        self.run = not hasattr(self.learn, 'lr_finder') and not hasattr(self, "gather_preds") and rank_distrib()==0
        if not self.run: return

        # Log config parameters
        log_config = self.learn.gather_args()
        _format_config(log_config)
        try:
            wandb.config.update(log_config, allow_val_change=True)
        except Exception as e:
            print(f'WandbCallback could not log config parameters -> {e}')

        if not WandbCallback._wandb_watch_called:
            WandbCallback._wandb_watch_called = True
            # Logs model topology and optionally gradients and weights
            wandb.watch(self.learn.model, log=self.log)

       
        # log dataset
        assert isinstance(self.log_dataset, (str, Path, bool)), 'log_dataset must be a path or a boolean'
        if self.log_dataset is True:
            if Path(self.dls.path) == Path('.'):
                print('WandbCallback could not retrieve the dataset path, please provide it explicitly to "log_dataset"')
                self.log_dataset = False
            else:
                self.log_dataset = self.dls.path
        
        if self.log_dataset:
            self.log_dataset = Path(self.log_dataset)
            assert self.log_dataset.is_dir(), f'log_dataset must be a valid directory: {self.log_dataset}'
            metadata = {'path relative to learner': os.path.relpath(self.log_dataset, self.learn.path)}
            if self.folder_names:
                assert isinstance(self.folder_names, list), 'folder_names must be a list of folder names as strings'
                for name in self.folder_names: assert isinstance(name, str), 'the elements of folder_names must be strings'
            log_dataset(main_path=self.log_dataset, folder_names=self.folder_names, name=self.dataset_name, metadata=metadata)


        # log model
        if self.log_model and not hasattr(self, 'save_model'):
            print('Adding SaveModelAtEndCallback()')
            self.learn.add_cb(SaveModelAtEndCallback())
            self.add_save_model = True
        else: self.add_save_model = False

        if self.log_preds:
            try:
                if not self.valid_dl:
                    if not len(self.dls.valid_ds):
                        print('Saving training set predictions')
                        #Initializes the batch watched
                        wandbRandom = random.Random(self.seed)  # For repeatability
                        self.n_preds = min(self.n_preds, len(self.dls.train_ds))
                        idxs = wandbRandom.sample(range(len(self.dls.train_ds)), self.n_preds)
                        test_items = [getattr(self.dls.train_ds.items, 'iloc', self.dls.train_ds.items)[i] for i in idxs]
                        self.preds_dl = self.dls.test_dl(test_items, with_labels=True)
                        
                else: self.preds_dl = self.valid_dl
                self.learn.add_cb(FetchPredsCallback(dl=self.preds_dl, with_input=True, with_decoded=True, reorder=self.reorder))
            except Exception as e:
                self.log_preds = False
                print(f'WandbCallback was not able to prepare a DataLoader for logging prediction samples -> {e}')

    def after_batch(self):
        "Log hyper-parameters and training loss"
        if self.training:
            self._wandb_step += 1
            self._wandb_epoch += 1/self.n_iter
            hypers = {f'{k}_{i}':v for i,h in enumerate(self.opt.hypers) for k,v in h.items()}

            wandb.log({'epoch': self._wandb_epoch, 'train_loss': float(to_detach(self.smooth_loss.clone())), 
                       'raw_loss': float(to_detach(self.loss.clone())), **hypers}, step=self._wandb_step)

    def log_predictions(self, preds):
        raise NotImplementedError("To be implemented")

    def after_epoch(self):
        "Log validation loss and custom metrics & log prediction samples"
        # Correct any epoch rounding error and overwrite value
        self._wandb_epoch = round(self._wandb_epoch)
        wandb.log({'epoch': self._wandb_epoch}, step=self._wandb_step)
        # Log sample predictions
        if self.log_preds:
            try:
                self.log_predictions(self.learn.fetch_preds.preds)
            except Exception as e:
                self.log_preds = False
                print(f'WandbCallback was not able to get prediction samples -> {e}')
        wandb.log({n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']}, step=self._wandb_step)

    def after_fit(self):
        if self.log_model:
            if self.save_model.last_saved_path is None:
                print('WandbCallback could not retrieve a model to upload')
            else:
                metadata = {n:s for n,s in zip(self.recorder.metric_names, self.recorder.log) if n not in ['train_loss', 'epoch', 'time']}
                log_model(self.save_model.last_saved_path, metadata=metadata)
        self.run = True
        self.learn.remove_cb(FetchPredsCallback)
        self.learn.remove_cb(SaveModelAtEndCallback)
        wandb.log({})  # ensure sync of last step
        self._wandb_step += 1
