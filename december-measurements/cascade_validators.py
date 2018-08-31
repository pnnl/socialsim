from functools import wraps


def check_root_only(default=None):
    """
    check if it is a single node cascade
    """
    def wrap(func):
        @wraps(func)
        def wrapped_f(self, *args, **kwargs):

            if len(self.main_df[self.main_df[self.node_col] != self.main_df[self.root_node_col]])==0:
                return default
            else:
                return func(self, *args, **kwargs)

        return wrapped_f

    return wrap
