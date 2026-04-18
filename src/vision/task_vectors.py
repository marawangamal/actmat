import os

import torch

from src.task_vectors import _TaskVector
from src.vision.linearize import LinearizedImageEncoder


class NonLinearTaskVector(_TaskVector):
    """A task vector for nonlinear models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return torch.load(checkpoint, map_location="cpu", weights_only=False)

    def apply_to_nonlinear(self, checkpoint_dir, scaling_coef=1.0):
        """Apply a task vector to a nonlinear pretrained model."""
        return self.apply_to(checkpoint_dir, scaling_coef)

    def apply_to_linear(self, checkpoint_dir, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return nonlinear_to_linear(self).apply_to(
            checkpoint_dir, scaling_coef
        )

    def _cast_to_same_type(self, other):
        return linear_to_nonlinear(other, self.vector.keys())

    def param_key_to_cov_key(self, key: str):
        return "image_encoder." + key.replace(".weight", "")


class LinearizedTaskVector(_TaskVector):
    """A task vector for linearized models."""

    def _load_checkpoint(self, checkpoint):
        """Load a checkpoint into a model."""
        return LinearizedImageEncoder.load(checkpoint)

    def apply_to_nonlinear(
        self, checkpoint_dir, param_names, scaling_coef=1.0
    ):
        """Apply a task vector to a nonlinear pretrained model."""
        return linear_to_nonlinear(self, param_names).apply_to(
            checkpoint_dir, scaling_coef
        )

    def apply_to_linear(self, checkpoint_dir, scaling_coef=1.0):
        """Apply a task vector to a linear pretrained model."""
        return self.apply_to(checkpoint_dir, scaling_coef)

    def get_named_parameters(self, param_names):
        """Get the named parameters of the task vector."""
        params = {k: v for k, v in self.vector.items() if "model.params0" not in k}
        return {k: v for k, v in zip(param_names, params.values())}

    def _cast_to_same_type(self, other):
        return nonlinear_to_linear(other)

    def param_key_to_cov_key(self, key: str):
        return "image_encoder." + key.replace(".weight", "")


class NonLinearWeightVector(NonLinearTaskVector):
    """Task vector that stores full finetuned weights rather than a delta.

    Used with ``--merge-mode=w``. Merging stacks full weights; ``apply_to``
    replaces the pretrained model's state_dict with ``self.vector``.
    """

    def _build_vector(self):
        assert not self.lazy, "NonLinearWeightVector does not support lazy mode"
        with torch.no_grad():
            pretrained = self._load_checkpoint(self._pretrained_checkpoint)
            finetuned = self._load_checkpoint(self._finetuned_checkpoint)
            p_sd = (
                pretrained.state_dict()
                if hasattr(pretrained, "state_dict")
                else pretrained
            )
            f_sd = (
                finetuned.state_dict()
                if hasattr(finetuned, "state_dict")
                else finetuned
            )
            vector, pretrained_sd = {}, {}
            for key in p_sd:
                if p_sd[key].dtype in (torch.int64, torch.uint8):
                    continue
                vector[key] = f_sd[key]
                pretrained_sd[key] = p_sd[key]
        self._pretrained_state_dict = pretrained_sd
        return vector

    def apply_to(self, checkpoint_dir, scaling_coef=1.0):
        pretrained_path = os.path.join(checkpoint_dir, self.PRETRAINED_FILENAME)
        with torch.no_grad():
            model = self._load_checkpoint(pretrained_path)
            sd = model.state_dict()
            new_sd = {
                k: (self.vector[k] if k in self.vector else sd[k]) for k in sd
            }
            model.load_state_dict(new_sd)
        return model

    def _copy_metadata(self, result):
        result = super()._copy_metadata(result)
        result._pretrained_state_dict = getattr(self, "_pretrained_state_dict", None)
        return result

    def map(self, fn):
        assert not self.lazy, "NonLinearWeightVector does not support lazy mode"
        with torch.no_grad():
            result = self.__class__(vector=fn(self.vector))
            result = self._copy_metadata(result)
            if getattr(self, "_pretrained_state_dict", None) is not None:
                result._pretrained_state_dict = fn(self._pretrained_state_dict)
        return result


class LinearizedWeightVector(LinearizedTaskVector):
    """Linearized counterpart of NonLinearWeightVector."""

    def _build_vector(self):
        assert not self.lazy, "LinearizedWeightVector does not support lazy mode"
        with torch.no_grad():
            pretrained = self._load_checkpoint(self._pretrained_checkpoint)
            finetuned = self._load_checkpoint(self._finetuned_checkpoint)
            p_sd = (
                pretrained.state_dict()
                if hasattr(pretrained, "state_dict")
                else pretrained
            )
            f_sd = (
                finetuned.state_dict()
                if hasattr(finetuned, "state_dict")
                else finetuned
            )
            vector, pretrained_sd = {}, {}
            for key in p_sd:
                if p_sd[key].dtype in (torch.int64, torch.uint8):
                    continue
                vector[key] = f_sd[key]
                pretrained_sd[key] = p_sd[key]
        self._pretrained_state_dict = pretrained_sd
        return vector

    def apply_to(self, checkpoint_dir, scaling_coef=1.0):
        pretrained_path = os.path.join(checkpoint_dir, self.PRETRAINED_FILENAME)
        with torch.no_grad():
            model = self._load_checkpoint(pretrained_path)
            sd = model.state_dict()
            new_sd = {
                k: (self.vector[k] if k in self.vector else sd[k]) for k in sd
            }
            model.load_state_dict(new_sd)
        return model

    def _copy_metadata(self, result):
        result = super()._copy_metadata(result)
        result._pretrained_state_dict = getattr(self, "_pretrained_state_dict", None)
        return result

    def map(self, fn):
        assert not self.lazy, "LinearizedWeightVector does not support lazy mode"
        with torch.no_grad():
            result = self.__class__(vector=fn(self.vector))
            result = self._copy_metadata(result)
            if getattr(self, "_pretrained_state_dict", None) is not None:
                result._pretrained_state_dict = fn(self._pretrained_state_dict)
        return result


def nonlinear_to_linear(nonlinear_task_vector):
    """Convert a nonlinear task vector to a linear task vector."""
    if isinstance(nonlinear_task_vector, LinearizedTaskVector):
        return nonlinear_task_vector
    else:
        linear_params = {
            f"model.params.{i}": v
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        # The diff of the init params of the linearized models are all zero.
        linear_params |= {
            f"model.params0.{i}": torch.zeros_like(v)
            for i, v in enumerate(nonlinear_task_vector.vector.values())
        }
        return LinearizedTaskVector(vector=linear_params)


def linear_to_nonlinear(linear_task_vector, param_names):
    """Convert a linear task vector to a nonlinear task vector."""
    if isinstance(linear_task_vector, NonLinearTaskVector):
        return linear_task_vector
    else:
        return NonLinearTaskVector(
            vector=linear_task_vector.get_named_parameters(param_names)
        )
