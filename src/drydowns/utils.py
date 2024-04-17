def is_true(setting):
    return setting.lower() in ["true", "yes", "1"]


def normalize(y, y_min, y_max):
    return (y - y_min) / (y_max - y_min)


def denormalize(y, y_min, y_max):
    return y * (y_max - y_min) + y_min


def normalize_param(a, y_min, y_max):
    return a / (y_max - y_min)


def denormalize_param(a, y_min, y_max):
    return a * (y_max - y_min)


def calc_k(ET_max, dz):
    return ET_max / dz


def calc_k(ET_max, dz):
    return ET_max / dz


def calc_delta_theta(theta_0, theta_w):
    return theta_0 - theta_w
