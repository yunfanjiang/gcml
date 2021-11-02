from abc import ABC, abstractmethod


class GCLBase(ABC):
    """
    Abstract base class for goal-conditioned learning, e.g., GCSL and GCML.
    """

    @abstractmethod
    def _compute_loss(self, *args, **kwargs):
        """
        Compute loss. It can be NLL loss for discrete action space or regression loss for continuous action space.
        This function will be invoked during training. It may also be invoked to calculate validation loss.
        """
        raise NotImplementedError

    @abstractmethod
    def _sample_trajectory(self, *args, **kwargs):
        """
        Generate a trajectory.
        """
        raise NotImplementedError

    @abstractmethod
    def _take_gradient_step(self, *args, **kwargs):
        """
        Take one gradient step to update parameters of the model.
        It corresponds to the policy update in GCSL. It also corresponds to the outer-loop step in GCML.
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, *args, **kwargs):
        """
        The main procedure for training.
        This method is exposed.
        """
        raise NotImplementedError

    @abstractmethod
    def test(self, *args, **kwargs):
        """
        Test method. It could correspond to the meta-test step in GCML.
        This method is exposed.
        """
        raise NotImplementedError
