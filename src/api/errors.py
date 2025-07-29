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