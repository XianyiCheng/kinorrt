import wrenchStampingLib as ws
import matlab.engine
from .mechanics import *
import numpy as np

def modes_to_int(modes):
    modes = np.array(modes)
    modes_int = np.zeros(modes.shape, dtype=int)
    modes_int[modes == CONTACT_MODE.LIFT_OFF] = 0
    modes_int[modes == CONTACT_MODE.FOLLOWING] = 1
    modes_int[modes == CONTACT_MODE.STICKING] = 1
    modes_int[modes == CONTACT_MODE.SLIDING_RIGHT] = 2
    modes_int[modes == CONTACT_MODE.SLIDING_LEFT] = 3
    return modes_int

class StabilityMarginSolver():
    def __init__(self):
        self.print_level=0 # 0: minimal screen outputs
        '''
        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('/home/xianyi/Documents/MATLAB/tbxmanager')
        #self.eng.startup(nargout=0)
        root = '/home/xianyi/libraries/prehensile_manipulation/matlab'
        self.eng.addpath(root)
        self.eng.addpath(root + '/utilities')
        self.eng.addpath(root + '/matlablibrary')
        self.eng.addpath(root + '/matlablibrary/Math/motion')
        self.eng.addpath(root + '/matlablibrary/Math/geometry')
        self.eng.addpath(root + '/matlablibrary/Math/array')
        self.eng.addpath(root + '/matlablibrary/Mesh/cone')
        self.eng.addpath('/home/xianyi/libraries/contact_mode_enumeration_2d')
        #self.eng.supress_warning(nargout=0)
        '''

    def compute_stablity_margin(self, env_mu, mnp_mu, envs, mnps, mode, obj_weight, mnp_fn_max, dist_weight):
        '''
        #geometry stability margin
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        dist_weight = dist_weight**0.5
        kCharacteristicLength = dist_weight

        CP_W_e = []
        CN_W_e = []
        for c in envs:
            CP_W_e.append(list(c.p))
            CN_W_e.append(list(c.n))
        CP_W_e = matlab.double(np.array(CP_W_e).T.tolist())
        CN_W_e = matlab.double(np.array(CN_W_e).T.tolist())

        CP_H_h = []
        CN_H_h = []
        for c in mnps:
            CP_H_h.append(list(c.p))
            CN_H_h.append(list(c.n))

        mode_number = []
        for m in mode:
            # 0:separation 1:fixed 2: right sliding 3: left sliding
            if m == CONTACT_MODE.LIFT_OFF:
                mode_number.append(0)
            elif m == CONTACT_MODE.FOLLOWING:
                mode_number.append(1)
            elif m == CONTACT_MODE.STICKING:
                mode_number.append(1)
            elif m == CONTACT_MODE.SLIDING_RIGHT:
                mode_number.append(2)
            elif m == CONTACT_MODE.SLIDING_LEFT:
                mode_number.append(3)

        CP_H_h = matlab.double(np.array(CP_H_h).T.tolist())
        CN_H_h = matlab.double(np.array(CN_H_h).T.tolist())
        CP_W_G = matlab.double([[0],[0]])
        R_WH = matlab.double([[1,0],[0,1]])
        p_WH = matlab.double([[0],[0]])
        h_mode = matlab.int8(mode_number[0:len(mnps)])
        e_mode = matlab.int8(mode_number[len(mnps):])

        score = self.eng.tryStabilityMargin(kFrictionE, kFrictionH, CP_W_e, CN_W_e, CP_H_h, CN_H_h, CP_W_G, R_WH,
                                                     p_WH, e_mode, h_mode, float(obj_weight), float(mnp_fn_max), kCharacteristicLength)
        '''
        score = 0
        return score

    def preprocess(self, x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes, obj_weight, mnp_fn_max, kCharacteristicLength = 0.15):
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        kContactForce = mnp_fn_max

        e_modes = modes_to_int(e_modes).astype('int32')
        h_modes = modes_to_int(h_modes).astype('int32')

        # Make contact info.
        Ad_gcos, depths, mus = contact_info(mnps, envs, mnp_mu, env_mu)
        N_e = []
        T_e = []
        N_h = []
        T_h = []
        eCone_allFix = []
        hCone_allFix = []
        n = np.array([[0.0], [1.0], [0.0]])
        D = np.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        n_c = len(Ad_gcos)
        for i in range(n_c):
            n_c = np.dot(Ad_gcos[i].T, n).T # 1x3
            t_c = np.dot(Ad_gcos[i].T, D).T# 2x3

            if i < len(mnps):
                f_c = n_c + t_c * kFrictionH
                f_c = f_c/np.linalg.norm(f_c[0])
                hCone_allFix.append(-f_c)
                N_h.append(-n_c[0])
                T_h.append(-t_c[0])
            else:
                f_c = n_c + t_c * kFrictionE
                f_c = f_c / np.linalg.norm(f_c[0])
                eCone_allFix.append(f_c)
                N_e.append(n_c[0])
                T_e.append(t_c[0])

        J_e = np.vstack((N_e, T_e))
        J_h = np.vstack((N_h, T_h))
        eCone_allFix = np.vstack(eCone_allFix)
        hCone_allFix = np.vstack(hCone_allFix)

        # Add gravity.
        f_g = obj_weight*np.array([[0.0], [-1.0], [0.0]])
        g_ow = np.linalg.inv(twist_to_transform(x))
        F_G = np.dot(g_ow, f_g)


        preprocess_params = (J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                             kContactForce, kFrictionE, kFrictionH,
                             kCharacteristicLength, e_modes, h_modes)

        return preprocess_params


    def stability_margin(self, preprocess_params, vo, mode):
        vo[abs(vo)<1e-5]=0
        J_e, J_h, eCone_allFix, hCone_allFix, \
        F_G,kContactForce, kFrictionE, kFrictionH,\
        kCharacteristicLength, e_modes, h_modes = preprocess_params

        mode = modes_to_int(mode).astype('int32')

        h_mode_goal = mode[0:h_modes.shape[1]].reshape(-1,1)
        e_mode_goal = mode[h_modes.shape[1]:].reshape(-1,1)

        G = np.zeros((3,6))
        G[0:3,0:3] = np.identity(3)
        b_G = np.array(vo).reshape(-1,1)

        stability_margin = ws.wrenchSpaceAnalysis_2d(J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                                                     kContactForce, kFrictionE, kFrictionH, kCharacteristicLength,
                                                     G, b_G, e_modes, h_modes, e_mode_goal, h_mode_goal, self.print_level)

        return stability_margin

    def test2d(self):

        kW = 0.0435  # object width
        kH = 0.0435  # object height
        env_mu = 0.3
        mnp_mu = 0.8

        h_modes = np.array([[CONTACT_MODE.STICKING, CONTACT_MODE.STICKING],
                            [CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.SLIDING_LEFT],
                            [CONTACT_MODE.STICKING, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING],
                            [CONTACT_MODE.SLIDING_LEFT, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.LIFT_OFF],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.SLIDING_RIGHT],
                            [CONTACT_MODE.LIFT_OFF, CONTACT_MODE.SLIDING_LEFT]])

        x = (0, 2.2, 0)
        # mnps = [Contact((kW/2, kH/2),(0,-1),0),Contact((-kW/2, kH/2),(0,-1),0)]
        mnps = [Contact((0.2097357615814568, 0.2), (0, -1), 0), Contact((-0.9389810887084302, 0.2), (0, -1), 0)]
        envs = [Contact((kW / 2, -kH / 2), (0, 1), 0), Contact((-kW / 2, -kH / 2), (0, 1), 0)]
        envs = [Contact((-1.0, -0.20000000000000018), (0, 1), 0), Contact((1.0, -0.20000000000000018), (0, 1), 0)]
        mode = [CONTACT_MODE.FOLLOWING, CONTACT_MODE.FOLLOWING, CONTACT_MODE.SLIDING_RIGHT, CONTACT_MODE.SLIDING_RIGHT]
        e_modes = np.array(get_contact_modes([], envs))
        e_modes = e_modes[~np.all(e_modes == CONTACT_MODE.LIFT_OFF, axis=1)]

        object_weight = 10
        mnp_fn_max = 15
        vel = [1, 0, 0]
        v = qp_inv_mechanics_2d(np.array(vel), np.array(x), mnps, envs, mode, 'vert', mnp_mu, env_mu, mnp_fn_max)

        preprocess = self.preprocess(x, env_mu, mnp_mu, envs, mnps, e_modes, h_modes,
                                         object_weight, mnp_fn_max)

        stability_margin_score = self.stability_margin(preprocess, v, mode)
        print(stability_margin_score)
        return stability_margin_score, preprocess, v, mode