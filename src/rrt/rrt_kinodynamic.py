import random
import numpy as np
from .rrt.rrt_base import RRTBase
from transformation import *
from itbl.mcp.qplcp import *
# import mechanics

class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4

class Contact(object):
    def __init__(self, position, normal, depth):
        self.p = tuple(position)
        self.n = tuple(normal)
        self.d = tuple(depth)

def smallfmod(x, y):
    while x > y:
        x -= y
    while x < 0:
        x += y
    return x

def v_hat(v):
    V = np.zeros([3,3])
    V[0:2, 0:2] = np.array([[0, -v[2]], [v[2], 0]])
    V[0:2, 2] = v[0:2].reshape(-1)
    return V

class RRTKinodynamic(RRTBase):

    def __init__(self, X, Q, x_init, x_goal, max_samples, r, prc=0.01):
        """
        Template RRTKinodynamic Planner
        """
        super().__init__(X, Q, x_init, x_goal, max_samples, r, prc)
        self.collision_manager = []
        self.mnp_mu = 0.8
        self.env_mu = 0.3

    def add_collision_manager(self, collision_manager, object):
        self.collision_manager = collision_manager
        self.object = object

    def set_transformation(self, x):
        T2 = config2trans(np.array(x))
        T3 = np.identity(4)
        T3[0:2,3] = T2[0:2,2]
        T3[0:2,0:2] = T2[0:2,0:2]
        self.object.transform()[:,:] = T3

    def check_collision(self, x):
        # return if_collide, contacts in object frame
        self.set_transformation(x)
        manifold = self.collision_manager.collide(self.object)

        n_pts = len(manifold.contact_points)
        contacts = []
        g_inv = inv_g_2d(config2trans(np.array(x)))

        for i in range(n_pts):
            cp = manifold.contact_points[i]
            cn = manifold.normals[i]
            cp_o = np.dot(g_inv,np.concatenate([cp,[1]]))
            cn_o = np.dot(g_inv[0:2,0:2], cn)
            ci = Contact(cp_o[0:2], cn_o, manifold.depths[i])
            contacts.append(ci)
        if_collide = len(manifold.depths)!=0

        return if_collide, contacts

    def check_penetration(self,contacts):
        ifPenetrate = False
        for c in contacts:
            if c.d < -0.01:
                ifCollide = True
                break
        return ifPenetrate

    def dist (self, p, q):
        cx = (p[0]-q[0])**2
        cy = (p[1]-q[1])**2
        period = 2*np.pi
        t1 = smallfmod(p[2], period)
        t2 = smallfmod(q[2], period)
        dt = t2-t1
        dt = smallfmod(dt + period/2.0, period) - period/2.0
        ct = dt**2
        return cx + cy + ct

    def inverse_mechanics(self, x, v_star, envs, mnps):
        v = inv_planar_2d(v_star, x, mnps, envs, self.mnp_mu, self.env_mu)
        # TODO: set friction coefficient here
        return v

    def forward_integration(self, x_near, x_rand, envs, mnps):
        # all collision checking, event detection , timestepping, ...
        h = 0.1
        x = x_near
        g_v = np.identity(3)
        g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]
        v_star = np.dot(g_v.T, steer(x, x_rand, h) - x)

        while np.linalg.norm(v_star) > 1e-2:

            v = inverse_mechanics(x, v_star, envs, mnps)

            # check collision
            if_collide, contacts = self.check_collision(x_)
            if not self.check_penetration(contacts):
                # update x if not collide
                x = x + np.dot(g_v, v)
                envs = contacts
                g_v = np.identity(3)
                g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]
                v_star = np.dot(g_v.T, steer(x, x_rand, h) - x)
            else:
                # return x backward at collisiton
                depths = np.array([c.d for c in contacts])
                max_i = np.argmax(depths)
                d_max = np.max(depths)
                p_max = contacts[max_i].p
                n_max = contacts[max_i].n
                vs = np.dot(adjointTrans_2d(config2trans(x)), v)
                v_p_max = np.dot(v_hat(dvs), np.concatenate([p_max, [1]]))
                k_new = 1 - d_max / abs(np.dot(v_p_max[0:2], n_max))
                if abs(k_new) < 1:
                    v = k_new * v
                    x = x + np.dot(g_v, v)
                break

        return tuple(x)

    def extend(self, x_near, x_rand, envs, mnps):
        v_star = steer(x_near, x_rand, self.Q[0])
        x_new = self.forward_integration(v, envs, mnps)
        return tuple(x_new)

    def connect(self, tree, x_rand):
        # nearest
        x_near = = self.get_nearest(0, x_rand)
        # collision check, get envs
        _, envs = self.check_collision(x_near)
        # manipulator contacts: sample from the previous one or new mnps
        mnps = []
        # extend
        x_new = self.extend(x_near, x_rand, envs, mnps)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 1e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        return x_new, status

    def search(self):

        self.add_vertex(0, self.x_init)
        self.add_edge(0, self.x_init, None)
        self.add_tree()

        self.connect(0, x_goal)

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            _, status = self.connect(0, x_rand)
            if status != Status.TRAPPED:
                _, goal_status = self.connect(0, x_goal)
                if goal_status == Status.REACHED:
                    path = self.reconstruct_path(0, self.x_init, self.x_goal)
                    return path

            self.samples_taken += 1
