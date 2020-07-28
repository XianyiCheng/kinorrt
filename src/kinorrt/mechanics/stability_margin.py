import matlab.engine
from .mechanics import *
import numpy as np

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
        kFrictionE = env_mu
        kFrictionH = mnp_mu
        dist_weight = dist_weight**0.5

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
                                                     p_WH, e_mode, h_mode, float(obj_weight), float(mnp_fn_max), dist_weight)
    

        return score
