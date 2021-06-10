import numpy as np
import torch


def log_standard_normal(x):
    return - 0.5 * x ** 2 - np.log(2 * np.pi) / 2


def sample_rademacher(shape):
    return (torch.rand(*shape).ge(0.5)).float() * 2 - 1


def sample_gaussian(shape):
    return torch.randn(*shape)


def sample_v(shape, vtype='rademacher'):
    if vtype == 'rademacher':
        return sample_rademacher(shape)
    elif vtype == 'normal' or vtype == 'gaussian':
        return sample_gaussian(shape)
    else:
        Exception(f'vtype {vtype} not supported')


Log2PI = float(np.log(2 * np.pi))


def log_normal(x, mean, log_var, eps=0.00001):
    z = - 0.5 * Log2PI
    return - (x - mean) ** 2 / (2. * torch.exp(log_var) + eps) - log_var / 2. + z


def exponential_CDF(t, lamb):
    return 1 - torch.exp(- lamb * t)


def sample_truncated_exponential(shape, lamb, T):
    """
    sample from q(t) prop lamb*exp(-lamb t) for t in [0, T]
    (one-sided truncation)
    """
    if lamb > 0:
        return - torch.log(1 - torch.rand(*shape).to(T) * exponential_CDF(T, lamb) + 1e-10) / lamb
    elif lamb == 0:
        return torch.rand(*shape).to(T) * T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')


def truncated_exponential_density(t, lamb, T):
    if lamb > 0:
        return lamb * torch.exp(-lamb * t) / exponential_CDF(T, lamb)
    elif lamb == 0:
        return 1 / T
    else:
        raise Exception(f'lamb must be nonnegative, got {lamb}')


def get_beta(iteration, anneal, beta_min=0.0, beta_max=1.0):
    assert anneal >= 1
    beta_range = beta_max - beta_min
    return min(beta_range * iteration / anneal + beta_min, beta_max)


def sample_ve_truncated_q(shape, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    return ve_truncated_q_inv_Phi(u.view(-1), sigma_min, sigma_max, t_epsilon, T).view(*shape)


def ve_truncated_q_density(t, sigma_min, sigma_max, t_epsilon, T):
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = np.log(r) * t_epsilon * m * r ** t_epsilon / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    gs2e = m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m) / A
    gs2 = m * r ** t * np.log(r) / (m * r ** t - m) / A

    return - torch.relu(- gs2 + gs2e) + gs2e


def ve_truncated_q_inv_Phi(u, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    # u = torch.rand(*shape).to(T)
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = t_epsilon * m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    # linear
    x_l = u * A * (m * r ** t_epsilon - m) / (m * r ** t_epsilon * np.log(r))
    mask = x_l.ge(t_epsilon).float()

    # nonlinear
    x_n = torch.log((r ** t_epsilon - 1) * torch.exp((A * u - A1)) + 1) / np.log(r)
    return mask * x_n + (1 - mask) * x_l


def ve_truncated_q_Phi(t, sigma_min, sigma_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    # u = torch.rand(*shape).to(T)
    m = sigma_min ** 2
    r = (sigma_max / sigma_min) ** 2
    A1 = t_epsilon * m * r ** t_epsilon * np.log(r) / (m * r ** t_epsilon - m)
    A2 = (torch.log(m * r ** T - m) - np.log(m * r ** t_epsilon - m))
    A = A1 + A2

    # linear
    u_l = t * (m * r ** t_epsilon * np.log(r)) / (A * (m * r ** t_epsilon - m))
    mask = t.ge(t_epsilon).float()

    # nonlinear
    u_n = A1 / A + (torch.log(m * r ** t - m) - np.log(m * r ** t_epsilon - m)) / A
    return mask * u_n + (1 - mask) * u_l


# noinspection PyTypeChecker
class VariancePreservingTruncatedSampling:

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20., t_epsilon=1e-3):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.t_epsilon = t_epsilon

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t

    def integral_beta(self, t):
        return 0.5 * t ** 2 * (self.beta_max - self.beta_min) + t * self.beta_min

    def mean_weight(self, t):
        # return torch.exp( -0.25 * t**2 * (self.beta_max-self.beta_min) - 0.5 * t * self.beta_min )
        return torch.exp(-0.5 * self.integral_beta(t))

    def var(self, t):
        # return 1. - torch.exp( -0.5 * t**2 * (self.beta_max-self.beta_min) - t * self.beta_min )
        return 1. - torch.exp(- self.integral_beta(t))

    def std(self, t):
        return self.var(t) ** 0.5

    def g(self, t):
        beta_t = self.beta(t)
        return beta_t ** 0.5

    def r(self, t):
        return self.beta(t) / self.var(t)

    def t_new(self, t):
        mask_le_t_eps = (t <= self.t_epsilon).float()
        t_new = mask_le_t_eps * t_eps + (1. - mask_le_t_eps) * t
        return t_new

    def unpdf(self, t):
        t_new = self.t_new(t)
        unprob = self.r(t_new)
        return unprob

    def antiderivative(self, t):
        return torch.log(1. - torch.exp(- self.integral_beta(t))) + self.integral_beta(t)

    def phi_t_le_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.r(t_eps).item() * t

    def phi_t_gt_t_eps(self, t):
        t_eps = torch.tensor(float(self.t_epsilon))
        return self.phi_t_le_t_eps(t_eps).item() + self.antiderivative(t) - self.antiderivative(t_eps).item()

    def normalizing_constant(self, T):
        return self.phi_t_gt_t_eps(T)

    def pdf(self, t, T):
        Z = self.normalizing_constant(T)
        prob = self.unpdf(t) / Z
        return prob

    def Phi(self, t, T):
        Z = self.normalizing_constant(T)
        t_new = self.t_new(t)
        mask_le_t_eps = (t <= self.t_epsilon).float()
        phi = mask_le_t_eps * self.phi_t_le_t_eps(t) + (1. - mask_le_t_eps) * self.phi_t_gt_t_eps(t_new)
        return phi / Z

    def inv_Phi(self, u, T):
        t_eps = torch.tensor(float(self.t_epsilon))
        Z = self.normalizing_constant(T)
        r_t_eps = self.r(t_eps).item()
        antdrv_t_eps = self.antiderivative(t_eps).item()
        mask_le_u_eps = (u <= self.t_epsilon * r_t_eps / Z).float()
        a = self.beta_max - self.beta_min
        b = self.beta_min
        inv_phi = mask_le_u_eps * Z / r_t_eps * u + (1. - mask_le_u_eps) * \
                  (-b + (b ** 2 + 2. * a * torch.log(
                      1. + torch.exp(Z * u + antdrv_t_eps - r_t_eps * self.t_epsilon))) ** 0.5) / a
        return inv_phi


# noinspection PyUnusedLocal
def sample_vp_truncated_q(shape, beta_min, beta_max, t_epsilon, T):
    if isinstance(T, float) or isinstance(T, int):
        T = torch.Tensor([T]).float()
    u = torch.rand(*shape).to(T)
    vpsde = VariancePreservingTruncatedSampling(beta_min=0.1, beta_max=20., t_epsilon=t_epsilon)
    return vpsde.inv_Phi(u.view(-1), T).view(*shape)
