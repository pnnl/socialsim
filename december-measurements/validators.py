from functools import wraps


def check_empty(default=None):
    def wrap(func):
        @wraps(func)
        def wrapped_f(self, *args, **kwargs):

            if self.main_df is None or self.main_df.empty or len(self.main_df) <= 0:
                return default
            else:
                return func(self, *args, **kwargs)

        return wrapped_f

    return wrap
