import matlab.engine
from .mechanics import *
import numpy as np
import wrenchStampingLib as ws

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

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath('/home/xianyi/MATLAB Add-Ons/Toolboxes/tbxmanager')
        #self.eng.addpath('/home/xianyi/Softwares/tbxmanager')
        self.eng.startup(nargout=0)
        root = '/home/xianyi/Dropbox/MLAB/PlanningThroughContact/ContactSeq_Planning/code/kinorrt/src/kinorrt/mechanics/prehensile_manipulation/matlab'
        self.eng.addpath(root)
        self.eng.addpath(root + '/utilities')
        self.eng.addpath(root + '/matlablibrary')
        self.eng.addpath(root + '/matlablibrary/Math/motion')
        self.eng.addpath(root + '/matlablibrary/Math/geometry')
        self.eng.addpath(root + '/matlablibrary/Math/array')
        self.eng.addpath(root + '/matlablibrary/Mesh/cone')
        self.eng.supress_warning(nargout=0)

    def compute_stablity_margin(self, env_mu, mnp_mu, envs, mnps, mode, obj_weight, mnp_fn_max, dist_weight):
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
    

        return score

    def preprocess0(self, env_mu, mnp_mu, envs, mnps, e_modes, h_modes, obj_weight, mnp_fn_max, kCharacteristicLength = 0.15):
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        kContactForce = mnp_fn_max
        kObjWeight = obj_weight
        kNumSlidingPlanes = 1  # for 2D planar problems

        CP_W_e = [] # env contact location
        CN_W_e = [] # env contact normal
        for c in envs:
            CP_W_e.append(list(c.p))
            CN_W_e.append(list(c.n))
        CP_W_e = np.array(CP_W_e).T
        CN_W_e = np.array(CN_W_e).T

        CP_H_h = [] # hand contact location
        CN_H_h = [] # hand contact normal
        for c in mnps:
            CP_H_h.append(list(c.p))
            CN_H_h.append(list(c.n))

        e_modes = modes_to_int(e_modes)
        h_modes = modes_to_int(h_modes)

        CP_H_h = np.array(CP_H_h).T
        CN_H_h = np.array(CN_H_h).T
        CP_W_G = np.array([[0], [0]])
        R_WH = np.array([[1, 0], [0, 1]])
        p_WH = np.array([[0], [0]])

        jacs = self.eng.preProcessing(matlab.double([kFrictionE]),
                                 matlab.double([kFrictionH]),
                                 matlab.double([kNumSlidingPlanes]),
                                 matlab.double([kObjWeight]),
                                 matlab.double(CP_W_e.tolist()),
                                 matlab.double(CN_W_e.tolist()),
                                 matlab.double(CP_H_h.tolist()),
                                 matlab.double(CN_H_h.tolist()),
                                 matlab.double(R_WH.tolist()),
                                 matlab.double(p_WH.tolist()),
                                 matlab.double(CP_W_G.tolist()), nargout=9)

        # read outputs
        N_e = np.asarray(jacs[0])
        T_e = np.asarray(jacs[1])
        N_h = np.asarray(jacs[2])
        T_h = np.asarray(jacs[3])
        eCone_allFix = np.asarray(jacs[4])
        hCone_allFix = np.asarray(jacs[6])
        F_G = np.asarray(jacs[8])

        J_e = np.vstack((N_e, T_e))
        J_h = np.vstack((N_h, T_h))

        preprocess_params = (J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                             kContactForce, kFrictionE, kFrictionH,
                             kCharacteristicLength, e_modes, h_modes)
        return preprocess_params

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
        D = np.array([[1.0, -1.0], [0.0, 0.0], [0.0, 0.0]])
        n_c = len(Ad_gcos)
        for i in range(n_c):
            n_c = np.dot(Ad_gcos[i].T, n).T # 1x3
            t_c = np.dot(Ad_gcos[i].T, D).T# 2x3

            if i < len(mnps):
                f_c = n_c + t_c * kFrictionH
                f_c = f_c/np.linalg.norm(f_c[0])
                hCone_allFix.append(f_c)
                N_h.append(n_c[0])
                T_h.append(t_c[0])
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
        J_e, J_h, eCone_allFix, hCone_allFix, \
        F_G,kContactForce, kFrictionE, kFrictionH,\
        kCharacteristicLength, e_modes, h_modes = preprocess_params

        mode = modes_to_int(mode).astype('int32')

        h_mode_goal = mode[0:h_modes.shape[1]].reshape(-1,1)
        e_mode_goal = mode[h_modes.shape[1]:].reshape(-1,1)

        G = np.zeros((3,6))
        G[0:3,0:3] = np.identity(3)
        b_G = np.array(vo).reshape(-1,1)

        print_level = 0  # 0: minimal screen outputs
        stability_margin = ws.wrenchSpaceAnalysis_2d(J_e, J_h, eCone_allFix, hCone_allFix, F_G,
                                                     kContactForce, kFrictionE, kFrictionH, kCharacteristicLength,
                                                     G, b_G, e_modes, h_modes, e_mode_goal, h_mode_goal, print_level)

        return stability_margin
