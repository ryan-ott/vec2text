from halo import Halo

def halo(text):
    def decorator(func):
        def wrapper(*args, **kwargs):
            spinner = Halo(text=text)
            spinner.start()
            try:
                result = func(*args, **kwargs)
                spinner.succeed(text=f"{text} - Success")
                return result
            except Exception as e:
                spinner.fail(text=f"{text} - Failed")
                raise e
        return wrapper
    return decorator
