"""
Tests for mentor.modalities — MentorModality, Classifier, Regressor.
"""
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from mentor.modalities import MentorModality, Classifier, Regressor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SimpleClassifier(Classifier):
    """Minimal Classifier subclass: one linear layer."""
    def __init__(self, in_features=8, num_classes=4):
        super().__init__(in_features=in_features, num_classes=num_classes)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)


class _SimpleRegressor(Regressor):
    """Minimal Regressor subclass: one linear layer -> scalar."""
    def __init__(self, in_features=8):
        super().__init__(in_features=in_features)
        self.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def _clf_batch(n=16, in_features=8, num_classes=4):
    x = torch.randn(n, in_features)
    y = torch.randint(0, num_classes, (n,))
    return x, y


def _reg_batch(n=16, in_features=8):
    x = torch.randn(n, in_features)
    y = torch.randn(n)
    return x, y


def _clf_loader(n=32, batch_size=8, in_features=8, num_classes=4):
    x = torch.randn(n, in_features)
    y = torch.randint(0, num_classes, (n,))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


def _reg_loader(n=32, batch_size=8, in_features=8):
    x = torch.randn(n, in_features)
    y = torch.randn(n)
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# MentorModality — abstract enforcement
# ---------------------------------------------------------------------------

class TestMentorModalityAbstract:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            MentorModality()

    def test_subclass_without_training_step_is_abstract(self):
        class _Incomplete(MentorModality):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
        with pytest.raises(TypeError):
            _Incomplete()

    def test_subclass_with_training_step_can_be_instantiated(self):
        class _Complete(MentorModality):
            def __init__(self):
                super().__init__()
                self.p = nn.Parameter(torch.tensor(1.0))
            def forward(self, x):
                return x * self.p
            def training_step(self, batch, loss_fn=None):
                x, y = batch
                loss = F.mse_loss(self(x), y.float())
                return loss, {"loss": loss.item()}
        m = _Complete()
        assert isinstance(m, MentorModality)

    def test_is_mentee_subclass(self):
        from mentor.mentee import Mentee
        assert issubclass(MentorModality, Mentee)

    def test_classifier_is_modality(self):
        assert issubclass(Classifier, MentorModality)

    def test_regressor_is_modality(self):
        assert issubclass(Regressor, MentorModality)


# ---------------------------------------------------------------------------
# Classifier — constructor param capture via MRO
# ---------------------------------------------------------------------------

class TestClassifierInit:
    def test_constructor_params_captured(self):
        m = _SimpleClassifier(in_features=8, num_classes=4)
        assert m._constructor_params == {"in_features": 8, "num_classes": 4}

    def test_constructor_params_non_default(self):
        m = _SimpleClassifier(in_features=16, num_classes=10)
        assert m._constructor_params == {"in_features": 16, "num_classes": 10}

    def test_default_loss_fn_initially_none(self):
        m = _SimpleClassifier()
        assert m._default_loss_fn is None

    def test_is_mentee_instance(self):
        from mentor.mentee import Mentee
        assert isinstance(_SimpleClassifier(), Mentee)


# ---------------------------------------------------------------------------
# Classifier — create_train_objects
# ---------------------------------------------------------------------------

class TestClassifierCreateTrainObjects:
    def test_returns_dict_with_required_keys(self):
        m = _SimpleClassifier()
        result = m.create_train_objects()
        assert {"optimizer", "lr_scheduler", "loss_fn"} <= set(result.keys())

    def test_sets_cross_entropy_as_default_loss(self):
        m = _SimpleClassifier()
        m.create_train_objects()
        assert isinstance(m._default_loss_fn, nn.CrossEntropyLoss)

    def test_loss_fn_in_dict_matches_default(self):
        m = _SimpleClassifier()
        result = m.create_train_objects()
        assert result["loss_fn"] is m._default_loss_fn

    def test_does_not_overwrite_existing_default_by_default(self):
        m = _SimpleClassifier()
        custom_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        m.create_train_objects(loss_fn=custom_loss)
        m.create_train_objects()  # second call — should not overwrite
        assert m._default_loss_fn is custom_loss

    def test_overwrites_when_flag_set(self):
        m = _SimpleClassifier()
        first_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        m.create_train_objects(loss_fn=first_loss)
        second_loss = nn.CrossEntropyLoss()
        m.create_train_objects(loss_fn=second_loss, overwrite_default_loss=True)
        assert m._default_loss_fn is second_loss

    def test_optimizer_type(self):
        m = _SimpleClassifier()
        result = m.create_train_objects()
        assert isinstance(result["optimizer"], torch.optim.Adam)

    def test_lr_scheduler_type(self):
        m = _SimpleClassifier()
        result = m.create_train_objects()
        assert isinstance(result["lr_scheduler"], torch.optim.lr_scheduler.StepLR)

    def test_custom_lr_applied(self):
        m = _SimpleClassifier()
        result = m.create_train_objects(lr=1e-4)
        opt = result["optimizer"]
        assert abs(opt.param_groups[0]["lr"] - 1e-4) < 1e-9

    def test_explicit_loss_fn_overrides_default(self):
        m = _SimpleClassifier()
        custom = nn.CrossEntropyLoss(label_smoothing=0.2)
        result = m.create_train_objects(loss_fn=custom)
        assert result["loss_fn"] is custom


# ---------------------------------------------------------------------------
# Classifier — training_step
# ---------------------------------------------------------------------------

class TestClassifierTrainingStep:
    def test_returns_tensor_and_dict(self):
        m = _SimpleClassifier()
        loss, metrics = m.training_step(_clf_batch())
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_loss_is_scalar(self):
        m = _SimpleClassifier()
        loss, _ = m.training_step(_clf_batch())
        assert loss.shape == torch.Size([])

    def test_loss_requires_grad(self):
        m = _SimpleClassifier()
        loss, _ = m.training_step(_clf_batch())
        assert loss.requires_grad

    def test_metrics_has_loss_and_acc(self):
        m = _SimpleClassifier()
        _, metrics = m.training_step(_clf_batch())
        assert "loss" in metrics
        assert "acc" in metrics

    def test_acc_in_zero_one(self):
        m = _SimpleClassifier()
        _, metrics = m.training_step(_clf_batch())
        assert 0.0 <= metrics["acc"] <= 1.0

    def test_uses_stateless_fallback_without_create_train_objects(self):
        # Should not raise even without calling create_train_objects first.
        m = _SimpleClassifier()
        loss, _ = m.training_step(_clf_batch())
        assert loss.item() >= 0.0

    def test_uses_default_loss_fn_when_set(self):
        m = _SimpleClassifier()
        m.create_train_objects()
        assert isinstance(m._default_loss_fn, nn.CrossEntropyLoss)
        # Should run without error using the registered default.
        loss, _ = m.training_step(_clf_batch())
        assert loss.item() >= 0.0

    def test_explicit_loss_fn_overrides_default(self):
        m = _SimpleClassifier()
        m.create_train_objects()  # sets CrossEntropyLoss
        called = []
        def custom_fn(logits, y):
            called.append(True)
            return F.cross_entropy(logits, y)
        loss, _ = m.training_step(_clf_batch(), loss_fn=custom_fn)
        assert called, "explicit loss_fn was not called"

    def test_first_metric_key_is_loss(self):
        m = _SimpleClassifier()
        _, metrics = m.training_step(_clf_batch())
        assert next(iter(metrics)) == "loss"


# ---------------------------------------------------------------------------
# Classifier — validation_step (delegates to training_step by default)
# ---------------------------------------------------------------------------

class TestClassifierValidationStep:
    def test_validation_step_returns_dict(self):
        m = _SimpleClassifier()
        with torch.no_grad():
            result = m.validation_step(_clf_batch())
        assert isinstance(result, dict)
        assert "loss" in result
        assert "acc" in result

    def test_validate_epoch_runs(self):
        m = _SimpleClassifier()
        loader = _clf_loader()
        metrics = m.validate_epoch(loader)
        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Classifier — full training epoch
# ---------------------------------------------------------------------------

class TestClassifierTrainEpoch:
    def test_train_epoch_runs(self):
        m = _SimpleClassifier()
        loader = _clf_loader()
        opt = m.create_train_objects()["optimizer"]
        metrics = m.train_epoch(loader, opt)
        assert "loss" in metrics
        assert "acc" in metrics

    def test_epoch_increments_current_epoch(self):
        m = _SimpleClassifier()
        loader = _clf_loader()
        opt = m.create_train_objects()["optimizer"]
        assert m.current_epoch == 0
        m.train_epoch(loader, opt)
        assert m.current_epoch == 1

    def test_weights_change_after_training(self):
        m = _SimpleClassifier()
        before = m.fc.weight.detach().clone()
        loader = _clf_loader()
        opt = m.create_train_objects()["optimizer"]
        m.train_epoch(loader, opt)
        assert not torch.equal(m.fc.weight.detach(), before)

    def test_save_and_resume(self, tmp_path):
        m = _SimpleClassifier(in_features=8, num_classes=4)
        loader = _clf_loader()
        opt = m.create_train_objects()["optimizer"]
        m.train_epoch(loader, opt)
        path = tmp_path / "clf.pt"
        m.save(str(path), optimizer=opt)

        from mentor.mentee import Mentee
        m2 = Mentee.resume(str(path), model_class=_SimpleClassifier)
        assert m2.current_epoch == 1
        assert m2._constructor_params == {"in_features": 8, "num_classes": 4}


# ---------------------------------------------------------------------------
# Regressor — constructor param capture
# ---------------------------------------------------------------------------

class TestRegressorInit:
    def test_constructor_params_captured(self):
        m = _SimpleRegressor(in_features=8)
        assert m._constructor_params == {"in_features": 8}

    def test_default_loss_fn_initially_none(self):
        m = _SimpleRegressor()
        assert m._default_loss_fn is None


# ---------------------------------------------------------------------------
# Regressor — create_train_objects
# ---------------------------------------------------------------------------

class TestRegressorCreateTrainObjects:
    def test_sets_mse_loss_as_default(self):
        m = _SimpleRegressor()
        m.create_train_objects()
        assert isinstance(m._default_loss_fn, nn.MSELoss)

    def test_returns_dict_with_required_keys(self):
        m = _SimpleRegressor()
        result = m.create_train_objects()
        assert {"optimizer", "lr_scheduler", "loss_fn"} <= set(result.keys())

    def test_does_not_overwrite_existing_default(self):
        m = _SimpleRegressor()
        custom = nn.L1Loss()
        m.create_train_objects(loss_fn=custom)
        m.create_train_objects()  # should keep custom
        assert m._default_loss_fn is custom

    def test_overwrites_when_flag_set(self):
        m = _SimpleRegressor()
        m.create_train_objects(loss_fn=nn.L1Loss())
        new_loss = nn.MSELoss()
        m.create_train_objects(loss_fn=new_loss, overwrite_default_loss=True)
        assert m._default_loss_fn is new_loss


# ---------------------------------------------------------------------------
# Regressor — training_step
# ---------------------------------------------------------------------------

class TestRegressorTrainingStep:
    def test_returns_tensor_and_dict(self):
        m = _SimpleRegressor()
        loss, metrics = m.training_step(_reg_batch())
        assert isinstance(loss, torch.Tensor)
        assert isinstance(metrics, dict)

    def test_loss_is_scalar(self):
        m = _SimpleRegressor()
        loss, _ = m.training_step(_reg_batch())
        assert loss.shape == torch.Size([])

    def test_loss_requires_grad(self):
        m = _SimpleRegressor()
        loss, _ = m.training_step(_reg_batch())
        assert loss.requires_grad

    def test_metrics_has_loss_and_rmse(self):
        m = _SimpleRegressor()
        _, metrics = m.training_step(_reg_batch())
        assert "loss" in metrics
        assert "rmse" in metrics

    def test_rmse_equals_sqrt_loss(self):
        m = _SimpleRegressor()
        _, metrics = m.training_step(_reg_batch())
        assert abs(metrics["rmse"] - metrics["loss"] ** 0.5) < 1e-6

    def test_integer_targets_accepted(self):
        # DataLoaders often yield integer targets; Regressor should handle them.
        m = _SimpleRegressor()
        x = torch.randn(8, 8)
        y = torch.randint(0, 10, (8,))  # integer targets
        loss, _ = m.training_step((x, y))
        assert loss.item() >= 0.0

    def test_stateless_fallback_without_create_train_objects(self):
        m = _SimpleRegressor()
        loss, _ = m.training_step(_reg_batch())
        assert loss.item() >= 0.0

    def test_explicit_loss_fn_overrides_default(self):
        m = _SimpleRegressor()
        m.create_train_objects()
        called = []
        def custom_fn(pred, target):
            called.append(True)
            return F.mse_loss(pred, target)
        m.training_step(_reg_batch(), loss_fn=custom_fn)
        assert called

    def test_first_metric_key_is_loss(self):
        m = _SimpleRegressor()
        _, metrics = m.training_step(_reg_batch())
        assert next(iter(metrics)) == "loss"


# ---------------------------------------------------------------------------
# Regressor — full training epoch
# ---------------------------------------------------------------------------

class TestRegressorTrainEpoch:
    def test_train_epoch_runs(self):
        m = _SimpleRegressor()
        loader = _reg_loader()
        opt = m.create_train_objects()["optimizer"]
        metrics = m.train_epoch(loader, opt)
        assert "loss" in metrics
        assert "rmse" in metrics

    def test_epoch_increments_current_epoch(self):
        m = _SimpleRegressor()
        loader = _reg_loader()
        opt = m.create_train_objects()["optimizer"]
        m.train_epoch(loader, opt)
        assert m.current_epoch == 1

    def test_save_and_resume(self, tmp_path):
        m = _SimpleRegressor(in_features=8)
        loader = _reg_loader()
        opt = m.create_train_objects()["optimizer"]
        m.train_epoch(loader, opt)
        path = tmp_path / "reg.pt"
        m.save(str(path), optimizer=opt)

        from mentor.mentee import Mentee
        m2 = Mentee.resume(str(path), model_class=_SimpleRegressor)
        assert m2.current_epoch == 1
        assert m2._constructor_params == {"in_features": 8}


# ---------------------------------------------------------------------------
# MRO sanity
# ---------------------------------------------------------------------------

class TestMRO:
    def test_classifier_mro_order(self):
        mro = _SimpleClassifier.__mro__
        names = [c.__name__ for c in mro]
        # Concrete class must precede modality, which must precede Mentee
        assert names.index("_SimpleClassifier") < names.index("Classifier")
        assert names.index("Classifier") < names.index("MentorModality")
        assert names.index("MentorModality") < names.index("Mentee")

    def test_regressor_mro_order(self):
        mro = _SimpleRegressor.__mro__
        names = [c.__name__ for c in mro]
        assert names.index("_SimpleRegressor") < names.index("Regressor")
        assert names.index("Regressor") < names.index("MentorModality")
        assert names.index("MentorModality") < names.index("Mentee")

    def test_isinstance_chain_classifier(self):
        m = _SimpleClassifier()
        from mentor.mentee import Mentee
        assert isinstance(m, _SimpleClassifier)
        assert isinstance(m, Classifier)
        assert isinstance(m, MentorModality)
        assert isinstance(m, Mentee)
        assert isinstance(m, nn.Module)

    def test_isinstance_chain_regressor(self):
        m = _SimpleRegressor()
        from mentor.mentee import Mentee
        assert isinstance(m, _SimpleRegressor)
        assert isinstance(m, Regressor)
        assert isinstance(m, MentorModality)
        assert isinstance(m, Mentee)
        assert isinstance(m, nn.Module)
