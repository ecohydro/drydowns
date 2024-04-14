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

def get_params(self, event, model='q', k=True, delta_theta=True, norm=True):
    q = np.array([
        0.,             # min_q
        np.inf,         # max_q
        1.0 + 1.0e-03   # ini_q
    ])

    et_max = np.array([
        0.,             # min_ET_max
        100.,           # max_ET_max
        event.pet       # ini_ET_max
    ])

    theta_0 = np.array([
        event.theta_w,  # min_theta_0
        # event.theta_star, # max_theta_0
        np.minimum(event.theta_star, self.data.theta_fc), # max_theta_0 
        # event.theta_star is either data.max_sm or data.theta_fc
        event.event_range + event.theta_w # ini_theta_0
    ])

    theta_star = np.array([
            0.0,            # min_theta_star
            self.data.theta_fc, # max_theta_star
            event.theta_star # ini_theta_star
        ])

    theta_w = event.theta_w

    k = et_max / (self.data.z*1000)
    delta_theta = theta_0 - event.theta_w


    if norm:
        theta_0 = utils.normalize(theta_0, y_min=self.theta_w, y_max=self.theta_star) # adds theta_w to numerator
        et_max = utils.normalize_param(et_max, y_min=self.theta_w, y_max=self.theta_star)

        k = utils.normalize_param(k, y_min=self.theta_w, y_max=self.theta_star)
        delta_theta = utils.normalize_param(delta_theta, y_min=self.theta_w, y_max=self.theta_star)

        theta_w, theta_star = (0.0, 1.0)
    
    