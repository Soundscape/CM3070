"""Modules providing utilities relating to the model"""

import json
import os
from logging import getLogger

import optuna
import pytorch_lightning as pl
import yaml
from optuna.pruners import SuccessiveHalvingPruner
from optuna.trial import TrialState

from models.classifiers import ScheduleClassifier, ScheduleDataModule


class ModelService:
    """Utility service for working with models
    """

    def __init__(self) -> None:
        games = 10000
        self.file_path = os.path.join(os.getcwd(), f'data/games_{games}.csv')
        self.logger = getLogger('tune')
        pl.seed_everything(42)

    def get_config(self) -> dict:
        """Loads hyper-parameters from disk

        Returns:
            dict: The hyper-parameters
        """
        with open('./trial_params.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    def set_config(self, params):
        """Saves hyper-parameters to disk

        Args:
            params (dict): The hyper-parameters
        """
        with open('./trial_params.json', 'w', encoding='utf-8') as f:
            json.dump(params, f)

    def tune(self, n_trials: int = 20, epochs: int = 10):
        """Tunes the model using ASHA

        Args:
            n_trials (int, optional): The number of trials. Defaults to 20.
            epochs (int, optional): The number of epochs. Defaults to 10.

        Returns:
            dict: The best hyper-parameters discovered in the trials
        """
        pruner = SuccessiveHalvingPruner(min_resource=1)
        study = optuna.create_study(direction="maximize", pruner=pruner)
        study.optimize(lambda trial: self._objective(
            trial, epochs), n_trials=n_trials)

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE])
        trial = study.best_trial

        self.logger.info(
            '\nStudy statistics:\n'
            f'\tNumber of finished trials: {len(study.trials)}\n'
            f'\tNumber of pruned trials: {len(pruned_trials)}\n'
            f'\tNumber of complete trials: {len(complete_trials)}\n'
        )

        msg = [
            '\nBest trial:\n',
            f'\tValue: {trial.value}\n',
            '\tParameters:\n'
        ]
        for key, value in trial.params.items():
            msg.append(f'\t\t{key}: {value}\n')
        self.logger.info(''.join(msg))

        self.set_config(trial.params)

        return trial.params

    def test(self, epochs: int = 10):
        """Trains the model using train, validation, and test sets. Indicates how the model performs on unseen data.

        Args:
            epochs (int, optional): The number of epochs to train. Defaults to 10.

        Returns:
            dict: The training metrics
        """
        config = self.get_config()

        classifier = ScheduleClassifier(config)
        data = ScheduleDataModule(config)

        classifier.load_from_checkpoint(
            './lightning_logs/version_8/checkpoints/epoch=14-step=5865.ckpt', config=config)
        trainer = pl.Trainer(accelerator='auto', max_epochs=epochs)

        for epoch in range(epochs):
            self.logger.info(f'Starting EPOCH {epoch+1}/{epochs}')
            trainer.fit(model=classifier, datamodule=data)

        trainer.test(model=classifier, datamodule=data)
        print(trainer.callback_metrics)

        trainer.save_checkpoint('./test_model.checkpoint')

        return trainer.callback_metrics

    def train(self, epochs: int = 10):
        """Trains the model on all data. Used to create the final model

        Args:
            epochs (int, optional): The number of epochs to train. Defaults to 10.

        Returns:
            dict: The training metrics
        """
        config = self.get_config()

        config["train_frac"] = 1.0
        config["test_frac"] = 0.0
        config["val_frac"] = 0.0

        classifier = ScheduleClassifier(config)
        data = ScheduleDataModule(config)

        classifier.load_from_checkpoint(
            './lightning_logs/version_8/checkpoints/epoch=14-step=5865.ckpt', config=config)
        trainer = pl.Trainer(accelerator='auto', max_epochs=epochs)

        for epoch in range(epochs):
            self.logger.info(f'Starting EPOCH {epoch+1}/{epochs}')
            trainer.fit(model=classifier, datamodule=data)

        trainer.save_checkpoint('./model.checkpoint')

        return trainer.callback_metrics

    def _create_classifier(self, trial):
        """Creates an instance of the classifier model for the current trial

        Args:
            trial (dict): The trial hyper-parameters

        Raises:
            optuna.TrialPruned: Indicates that a trial should be pruned.

        Returns:
            (classifier, data): The classifier and its data module
        """
        config = {
            "lr": trial.suggest_categorical("lr", [1e-3, 1e-4, 1e-5, 1e-6]),
            "input_dim": trial.suggest_categorical("input_dim", [120]),
            "output_dim": trial.suggest_categorical("output_dim", [96]),
            "fc1_dim": trial.suggest_categorical("fc1_dim", [256, 512, 1024]),
            "fc2_dim": trial.suggest_categorical("fc2_dim", [256, 512, 1024]),
            "fc3_dim": trial.suggest_categorical("fc3_dim", [256, 512, 1024]),
            "fc4_dim": trial.suggest_categorical("fc4_dim", [256, 512, 1024]),
            "fc5_dim": trial.suggest_categorical("fc5_dim", [256, 512, 1024]),
            "fc6_dim": trial.suggest_categorical("fc6_dim", [256, 512, 1024]),
            "dropout": trial.suggest_categorical("dropout", [0.1, 0.2]),
            "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            "train_frac": trial.suggest_categorical("train_frac", [0.8]),
            "test_frac": trial.suggest_categorical("test_frac", [0.1]),
            "val_frac": trial.suggest_categorical("val_frac", [0.1]),
            "num_workers": trial.suggest_categorical("num_workers", [8]),
            "file_path": trial.suggest_categorical("file_path", [self.file_path])
        }

        if trial.should_prune():
            raise optuna.TrialPruned()

        classifier = ScheduleClassifier(config)
        data = ScheduleDataModule(config)

        return classifier, data

    def _objective(self, trial, epochs: int = 10):
        """Objective function used to determine which hyper-parameters are performing best

        Args:
            trial (dict): The current hyper-parameters for the trial
            epochs (int, optional): The number of epochs. Defaults to 10.

        Returns:
            float: Validation accuracy
        """
        classifier, data = self._create_classifier(trial)

        trainer = pl.Trainer(accelerator='auto', max_epochs=epochs, callbacks=[
                             OnCheckpointHparams(trial)])
        trainer.fit(model=classifier, datamodule=data)

        keys = trial.params.keys()
        msg = 'Trial parameters:\n' + \
            ''.join([f'\t{key}: {trial.params[key]}\n' for key in keys])
        self.logger.info(msg)

        return trainer.callback_metrics["val_accuracy"]


class OnCheckpointHparams(pl.Callback):
    """Persists hyper parameters to disk when checkpointing
    """

    def __init__(self, trial) -> None:
        super().__init__()
        self.trial = trial

    def on_save_checkpoint(self, trainer, module, state):
        if trainer.current_epoch == 0:
            file_path = f"{trainer.logger.log_dir}/hparams.yaml"
            print(f"Saving hparams to file_path: {file_path}")
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.trial.params, f, default_flow_style=False)
