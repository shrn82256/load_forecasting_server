def get_db(device):
    return "db/{}.csv".format(device)


def exception_as_dict(ex):
    # return dict(type=ex.__class__.__name__,
    #             errno=ex.errno, message=ex.message,
    #             strerror=exception_as_dict(ex.strerror)
    #             if isinstance(ex.strerror, Exception) else ex.strerror)
    return str(ex)