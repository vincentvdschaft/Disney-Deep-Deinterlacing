"""
This module contains a callback handling system.

The CallbackHandler is intialized with a list of Callback objects, each
containing their respective trigger moments (a trigger moment is refered to as a
`when`). The CallbackHandler creates a callable attribute for every trigger
moment. This means that if there is a callback with trigger moment 'on_loop_end'
the attribute `callbackhandler.on_loop_end` will be initialized which can be
called with `callbackhandler.on_loop_end()`.
"""


class CallbackHandler:
    """Callback handler class that is used to call callbacks."""

    def __init__(self, callbacks):
        """
        Initializes a CallbackHandler object and creates callable objects for all
        detected trigger moments (whens).
        """
        assert isinstance(
            callbacks, list), 'callbacks must be a list of callbacks'

        # Add all whens referred to by the input callbacks to
        # the set of existing whens for this callback handler
        self.whens = set()
        for callback in callbacks:
            for when in callback.whens:
                self.whens.add(when)

        # Create a dict linking each when to the callbacks that should trigger
        self.callbacks = {when: [cb for cb in callbacks if when in cb.whens]
                          for when in self.whens}

        # Create a callable object for every when that calls all corresponding
        # callbacks This way the user could for example call the callback using
        # callback_handler.before_start()
        for when in self.whens:
            setattr(self, when, CallbackCallable(self.callbacks[when], when))


class CallbackCallable:
    """Callable object to be created by CallbackHandler for every trigger
    moment (when)."""

    def __init__(self, callbacks, when):
        """Initializes a callable object with a list of callbacks."""
        # Ensure all callbacks are actually BaseCallback objects
        assert all([isinstance(cb, BaseCallback) for cb in callbacks])
        # Store the callbacks and when
        self.callbacks = callbacks
        self.when = when

    def __call__(self, **kwargs):
        """Calls all stored callbacks."""
        for cb in self.callbacks:
            cb(when=self.when, **kwargs)


class BaseCallback:
    """Base class from which callback objects are derived."""

    def __init__(self, whens):
        """Initializes a callback with a list of trigger moments (whens)."""
        # If when is a single string, put it in a list
        if not isinstance(whens, list):
            whens = [whens]

        self.whens = whens

    def __call__(self, when, **kwargs):
        """Prints when and arguments."""
        if self._trigger_condition(when, **kwargs) is True:
            self._trigger(when, **kwargs)

    def _trigger_condition(self, when, **kwargs):
        """Extra condition to determine if callback triggers."""
        return True

    def _trigger(self, when, **kwargs):
        """Triggers the callback."""
        print(
            f'BaseCallback was called at trigger {when} with keyword arguments {kwargs}')


class BasePeriodicCallback(BaseCallback):
    def __init__(
            self,
            period,
            periodic_whens,
            unconditional_whens):
        """
        Initializes a callback that only triggers every n iterations. Triggers
        on n-1, 2xn-1, 3xn-1, etc.
        Args:
            periodic_whens : list(str)
                Whens that trigger only after n calls. (`iteration` must be
                supplied)
            period : int
                The number of calls before trigger
            unconditional_whens : list(str)
                The whens that always trigger the callback (might be used for a
                'loop_end' callback for instance)
        """
        if periodic_whens is None:
            periodic_whens = []

        if unconditional_whens is None:
            unconditional_whens = []

        # If when is a single string, put it in a list
        if not isinstance(periodic_whens, list):
            periodic_whens = [periodic_whens]
        if not isinstance(unconditional_whens, list):
            unconditional_whens = [unconditional_whens]

        super().__init__([*periodic_whens, *unconditional_whens])
        self.every_n_whens = periodic_whens
        self.unconditional_whens = unconditional_whens

        # The amount of calls before the callback is triggered
        self.period = period

    def _trigger_condition(self, when, iteration=None, **kwargs):
        # Always trigger on the always_trigger_whens
        if when in self.unconditional_whens:
            return True

        # Trigger periodically
        if iteration is not None and iteration % self.period == 0 and iteration > 0:
            return True
        return False

    def _trigger(self, when, **kwargs):
        """Prints when and arguments at trigger."""
        print(
            'BasePeriodicCallback was called on trigger'
            f'{when} with keyword arguments {kwargs}')
