"""
Tests for mentor.trainers — MentorTrainer, Classifier, Regressor.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mentor.mentee import Mentee
from mentor.trainers import MentorTrainer, Classifier, Regressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleClassifier(Mentee):
    def __init__(self, in_features=8, num_classes=4):
        super().__init__(in_features=in_features, num_classes=num_classes)
        self.fc = nn.Linear(in_features, num_classes)
        self.trainer = Classifier()

    def forward(self, x):
        return self.fc(x)


class _SimpleRegressor(Mentee):
    def __init__(self, in_features=8):
        super().__init__(in_features=in_features)
        self.fc = nn.Linear(in_features, 1)
        self.trainer = Regressor()

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def _clf_batch(n=16, in_features=8, num_classes=4):
    return torch.randn(n, in_features), torch.randint(0, num_classes, (n,))


def _reg_batch(n=16, in_features=8):
    return torch.randn(n, in_features), torch.randn(n)


def _clf_loader(n=32, batch_size=8, in_features=8, num_classes=4):
    x, y = torch.randn(n, in_features), torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _reg_loader(n=32, batch_size=8, in_features=8):
    x, y = torch.randn(n, in_features), torch.randn(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# MentorTrainer — abstract enforcement
# ---------------------------------------------------------------------------

class TestMentorTrainerAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MentorTrainer()

    def test_incomplete_subclass_is_abstract(self):
        class _Bad(MentorTrainer):
            # missing default_training_step and create_train_objects
            pass
        with pytest.raises(TypeError):
            _Bad()

    def test_complete_subclass_can_be_instantiated(self):
        class _Good(MentorTrainer):
            @classmethod
            def default_training_step(cls, model, batch, loss_fn=None):
                x, y = batch
                loss = F.mse_loss(model(x), y.float())
                return loss, {"loss": loss.item()}
            def create_train_objects(self, model, lr=1e-3, step_size=10,
                                     gamma=0.1, loss_fn=None, overwrite_default_loss=False):
                self._optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                self._lr_scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, step_size, gamma)
                return {"optimizer": self._optimizer, "lr_scheduler": self._lr_scheduler, "loss_fn": None}
        t = _Good()
        assert isinstance(t, MentorTrainer)

    def test_is_not_nn_module(self):
        assert not issubclass(MentorTrainer, nn.Module)
        assert not issubclass(Classifier, nn.Module)
        assert not issubclass(Regressor, nn.Module)

    def test_is_not_mentee(self):
        assert not issubclass(MentorTrainer, Mentee)
        assert not issubclass(Classifier, Mentee)
        assert not issubclass(Regressor, Mentee)


# ---------------------------------------------------------------------------
# Trainer state — properties before and after create_train_objects
# ---------------------------------------------------------------------------

class TestTrainerProperties:
    def test_all_none_before_create(self):
        t = Classifier()
        assert t.optimizer is None
        assert t.lr_scheduler is None
        assert t.loss_fn is None

    def test_set_after_create(self):
        m = _SimpleClassifier()
        m.create_train_objects(lr=1e-3)
        assert isinstance(m.trainer.optimizer, torch.optim.Adam)
        assert isinstance(m.trainer.lr_scheduler, torch.optim.lr_scheduler.StepLR)
        assert isinstance(m.trainer.loss_fn, nn.CrossEntropyLoss)

    def test_trainer_properties_match_mentee_properties(self):
        m = _SimpleClassifier()
        m.create_train_objects()
        assert m.optimizer is m.trainer.optimizer
        assert m.lr_scheduler is m.trainer.lr_scheduler
        assert m.loss_fn is m.trainer.loss_fn

    def test_second_create_replaces_cached_objects(self):
        m = _SimpleClassifier()
        m.create_train_objects(lr=1e-3)
        first_opt = m.trainer.optimizer
        m.create_train_objects(lr=1e-4)
        assert m.trainer.optimizer is not first_opt
        assert abs(m.trainer.optimizer.param_groups[0]["lr"] - 1e-4) < 1e-9


# ---------------------------------------------------------------------------
# Mentee.trainer composition
# ---------------------------------------------------------------------------

class TestMenteeTrainerComposition:
    def test_trainer_none_by_default(self):
        class _Plain(Mentee):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
        m = _Plain()
        assert m.trainer is None

    def test_trainer_assignable_after_construction(self):
        class _Plain(Mentee):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
            def forward(self, x): return x * self.p
        m = _Plain()
        m.trainer = Classifier()
        assert isinstance(m.trainer, Classifier)

    def test_mentee_properties_none_without_trainer_and_without_create(self):
        m = _SimpleClassifier()
        m.trainer = None
        assert m.optimizer is None
        assert m.lr_scheduler is None
        assert m.loss_fn is None

    def test_create_train_objects_delegates_to_trainer(self):
        m = _SimpleClassifier()
        result = m.create_train_objects(lr=5e-4)
        assert result["optimizer"] is m.trainer.optimizer
        assert abs(m.trainer.optimizer.param_groups[0]["lr"] - 5e-4) < 1e-9

    def test_training_step_raises_without_trainer_and_no_override(self):
        class _NoTrainer(Mentee):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
        m = _NoTrainer()
        with pytest.raises(NotImplementedError):
            m.training_step((torch.tensor([1.0]), torch.tensor([1.0])))

    def test_training_step_delegates_to_trainer(self):
        m = _SimpleClassifier()
        loss, metrics = m.training_step(_clf_batch())
        assert isinstance(loss, torch.Tensor)
        assert "loss" in metrics and "acc" in metrics

    def test_validation_step_delegates_to_trainer(self):
        m = _SimpleClassifier()
        with torch.no_grad():
            metrics = m.validation_step(_clf_batch())
        assert isinstance(metrics, dict)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Classifier trainer
# ---------------------------------------------------------------------------

class TestClassifier:
    def test_training_step_returns_tensor_and_dict(self):
        m = _SimpleClassifier()
        loss, metrics = m.training_step(_clf_batch())
        assert isinstance(loss, torch.Tensor) and loss.shape == torch.Size([])
        assert isinstance(metrics, dict)

    def test_metrics_keys(self):
        m = _SimpleClassifier()
        _, metrics = m.training_step(_clf_batch())
        assert "loss" in metrics and "acc" in metrics

    def test_acc_in_range(self):
        m = _SimpleClassifier()
        _, metrics = m.training_step(_clf_batch())
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_loss_requires_grad(self):
        m = _SimpleClassifier()
        loss, _ = m.training_step(_clf_batch())
        assert loss.requires_grad

    def test_stateless_fallback_no_create(self):
        m = _SimpleClassifier()
        loss, _ = m.training_step(_clf_batch())
        assert loss.item() >= 0.0

    def test_creates_cross_entropy_default(self):
        m = _SimpleClassifier()
        m.create_train_objects()
        assert isinstance(m.loss_fn, nn.CrossEntropyLoss)

    def test_explicit_loss_fn_used(self):
        m = _SimpleClassifier()
        called = []
        def custom(logits, y):
            called.append(True)
            return F.cross_entropy(logits, y)
        m.training_step(_clf_batch(), loss_fn=custom)
        assert called

    def test_does_not_overwrite_existing_loss_by_default(self):
        m = _SimpleClassifier()
        custom = nn.CrossEntropyLoss(label_smoothing=0.1)
        m.create_train_objects(loss_fn=custom)
        m.create_train_objects()
        assert m.trainer.loss_fn is custom

    def test_overwrites_when_flag_set(self):
        m = _SimpleClassifier()
        m.create_train_objects(loss_fn=nn.CrossEntropyLoss(label_smoothing=0.1))
        new_loss = nn.CrossEntropyLoss()
        m.create_train_objects(loss_fn=new_loss, overwrite_default_loss=True)
        assert m.trainer.loss_fn is new_loss

    def test_train_epoch_runs(self):
        m = _SimpleClassifier()
        opt = m.create_train_objects()["optimizer"]
        metrics = m.train_epoch(_clf_loader(), opt)
        assert "loss" in metrics and "acc" in metrics
        assert m.current_epoch == 1

    def test_validate_epoch_runs(self):
        m = _SimpleClassifier()
        opt = m.create_train_objects()["optimizer"]
        m.train_epoch(_clf_loader(), opt)
        metrics = m.validate_epoch(_clf_loader())
        assert "loss" in metrics

    def test_save_and_resume(self, tmp_path):
        m = _SimpleClassifier(in_features=8, num_classes=4)
        m.create_train_objects()
        m.train_epoch(_clf_loader(), m.optimizer)
        path = tmp_path / "clf.pt"
        m.save(str(path))
        m2 = Mentee.resume(str(path), model_class=_SimpleClassifier)
        assert m2.current_epoch == 1
        assert m2._constructor_params == {"in_features": 8, "num_classes": 4}


# ---------------------------------------------------------------------------
# Regressor trainer
# ---------------------------------------------------------------------------

class TestRegressor:
    def test_training_step_returns_tensor_and_dict(self):
        m = _SimpleRegressor()
        loss, metrics = m.training_step(_reg_batch())
        assert isinstance(loss, torch.Tensor) and loss.shape == torch.Size([])

    def test_metrics_keys(self):
        m = _SimpleRegressor()
        _, metrics = m.training_step(_reg_batch())
        assert "loss" in metrics and "rmse" in metrics

    def test_rmse_equals_sqrt_loss(self):
        m = _SimpleRegressor()
        _, metrics = m.training_step(_reg_batch())
        assert abs(metrics["rmse"] - metrics["loss"] ** 0.5) < 1e-6

    def test_integer_targets_accepted(self):
        m = _SimpleRegressor()
        x = torch.randn(8, 8)
        y = torch.randint(0, 10, (8,))
        loss, _ = m.training_step((x, y))
        assert loss.item() >= 0.0

    def test_creates_mse_default(self):
        m = _SimpleRegressor()
        m.create_train_objects()
        assert isinstance(m.loss_fn, nn.MSELoss)

    def test_train_epoch_runs(self):
        m = _SimpleRegressor()
        opt = m.create_train_objects()["optimizer"]
        metrics = m.train_epoch(_reg_loader(), opt)
        assert "loss" in metrics and "rmse" in metrics

    def test_save_and_resume(self, tmp_path):
        m = _SimpleRegressor(in_features=8)
        m.create_train_objects()
        m.train_epoch(_reg_loader(), m.optimizer)
        path = tmp_path / "reg.pt"
        m.save(str(path))
        m2 = Mentee.resume(str(path), model_class=_SimpleRegressor)
        assert m2.current_epoch == 1
        assert m2._constructor_params == {"in_features": 8}


# ---------------------------------------------------------------------------
# Cached properties on Mentee (via trainer)
# ---------------------------------------------------------------------------

class TestCachedProperties:
    def test_optimizer_none_before_create(self):
        assert _SimpleClassifier().optimizer is None

    def test_lr_scheduler_none_before_create(self):
        assert _SimpleClassifier().lr_scheduler is None

    def test_loss_fn_none_before_create(self):
        assert _SimpleClassifier().loss_fn is None

    def test_all_set_after_create(self):
        m = _SimpleClassifier()
        m.create_train_objects()
        assert isinstance(m.optimizer, torch.optim.Adam)
        assert isinstance(m.lr_scheduler, torch.optim.lr_scheduler.StepLR)
        assert isinstance(m.loss_fn, nn.CrossEntropyLoss)

    def test_save_auto_uses_cached_optimizer(self, tmp_path):
        m = _SimpleClassifier()
        m.create_train_objects()
        m.train_epoch(_clf_loader(), m.optimizer)
        path = tmp_path / "auto.pt"
        m.save(str(path))
        cp = torch.load(str(path), weights_only=False)
        assert "optimizer_state" in cp
        assert "lr_scheduler_state" in cp

    def test_regressor_cached_properties(self):
        m = _SimpleRegressor()
        m.create_train_objects()
        assert isinstance(m.optimizer, torch.optim.Adam)
        assert isinstance(m.lr_scheduler, torch.optim.lr_scheduler.StepLR)
        assert isinstance(m.loss_fn, nn.MSELoss)
