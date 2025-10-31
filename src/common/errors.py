class BadRequestError(Exception):
    """Exception raised for bad requests.

    NOTE: throwing this will result in a 400 response with the given error message.
    """

    def __init__(self, message):
        self.message = message

class MissingResourceError(Exception):
    """Exception raised when a requested resource is not found.

    NOTE: throwing this will result in a 404 response with the given error message.
    """

    def __init__(self, message):
        self.message = message

class TaggerRuntimeError(Exception):
    """Base exception with context"""
    def __init__(self, message: str, **context):
        super().__init__(message)
        self.context = context
    
    def __str__(self):
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{super().__str__()} ({context_str})"
        return super().__str__()