import random
import numpy as np
import enum
from .rrt_base import RRTBase
from ..utilities.transformations import *
from ..utilities.geometry import steer
from ..mechanics.mechanics import *
from ..mechanics.stability_margin import *
from .tree import RRTTree, RRTEdge
import time

class Status(enum.Enum):
    FAILED = 1
    TRAPPED = 2
    ADVANCED = 3
    REACHED = 4

def smallfmod(x, y):
    while x > y:
        x -= y
    while x < 0:
        x += y
    return x


class RRTKinodynamic(RRTBase):

    def __init__(self, X, x_init, x_goal, envir, object, manipulator, max_samples, r = 5, world = 'planar'):
        """
        Template RRTKinodynamic Planner
        """
        self.X = X
        self.samples_taken = 0
        self.max_samples = max_samples
        self.x_init = x_init
        self.x_goal = x_goal
        self.neighbor_radius = r
        self.world = world
        self.trees = []  # list of all trees
        self.add_tree()  # add initial tree
        self.environment = envir
        self.object = object
        self.manipulator = manipulator

        self.collision_manager = []
        self.mnp_mu = 0.8
        self.env_mu = 0.3
        self.dist_weight = 1
        self.goal_kch = [1,1,1]
        self.cost_weight = [0.2,1,1]
        self.step_length = 2
        # self.trees_node = [[]] # TODO: hack here
        self.mnp_fn_max = None

        self.swapped = False

    def set_world(self, key):
        self.world = key

    def get_nearest(self, tree, x):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """
        min_d = np.inf
        for q in self.trees[tree].nodes:
            d = self.dist(q, x)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near

    def get_unexpand_nearest(self, tree):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """

        min_d = np.inf
        q_near = self.x_init
        for q in self.trees[tree].nodes:
            if self.trees[tree].goal_expand[q]:
                continue
            d = self.goal_dist(q)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near

    def get_goal_nearest(self, tree):
        """
        Return vertex nearest to x
        :param tree: int, tree being searched
        :param x: tuple, vertex around which searching
        :return: tuple, nearest vertex to x
        """

        min_d = np.inf
        for q in self.trees[tree].nodes:
            d = self.goal_dist(q)
            if q in self.trees[tree].edges:
                if min_d > d:
                    min_d = d
                    q_near = q
            else:
                self.trees[tree].nodes.remove(q)

        return q_near, min_d

    def add_collision_manager(self, collision_manager, object, object_shape):
        self.collision_manager = collision_manager
        self.object = object
        self.object_shape = object_shape

    def check_collision(self, x):
        if_collide, w_contacts = self.environment.check_collision(self.object, x)
        contacts = self.object.contacts2objframe(w_contacts, x)
        #
        # # return if_collide, contacts in object frame
        # T2 = config2trans(np.array(x))
        # T3 = np.identity(4)
        # T3[0:2, 3] = T2[0:2, 2]
        # T3[0:2, 0:2] = T2[0:2, 0:2]
        # self.object.transform()[:, :] = T3
        # manifold = self.collision_manager.collide(self.object)
        #
        # n_pts = len(manifold.contact_points)
        # contacts = []
        # g_inv = inv_g_2d(config2trans(np.array(x)))
        #
        # # the contacts are wrt the object frame
        # for i in range(n_pts):
        #     cp = manifold.contact_points[i]
        #     cn = manifold.normals[i]
        #     cp_o = np.dot(g_inv,np.concatenate([cp,[1]]))
        #     cn_o = np.dot(g_inv[0:2,0:2], cn)
        #     ci = Contact(cp_o[0:2], cn_o, manifold.depths[i])
        #     contacts.append(ci)
        # if_collide = len(manifold.depths) != 0

        return if_collide, contacts

    def check_penetration(self,contacts):
        ifPenetrate = False
        for c in contacts:
            if c.d < -0.05:
                ifPenetrate = True
                #print('penetrate')
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
        ct = self.dist_weight*dt**2
        return cx + cy + ct
    
    def goal_dist(self, p):
        q = self.x_goal
        cx = (p[0] - q[0]) ** 2
        cy = (p[1] - q[1]) ** 2
        period = 2 * np.pi
        t1 = smallfmod(p[2], period)
        t2 = smallfmod(q[2], period)
        dt = t2 - t1
        dt = smallfmod(dt + period / 2.0, period) - period / 2.0
        ct = dt ** 2
        dist = self.goal_kch[0]*cx + self.goal_kch[1]*cy + self.goal_kch[2]*ct
        return dist

    def swap_trees(self):
        """
        Swap trees only
        """
        # swap trees
        self.trees[0], self.trees[1] = self.trees[1], self.trees[0]
        self.swapped = not self.swapped

    def unswap(self):
        """
        Check if trees have been swapped and unswap
        """
        if self.swapped:
            self.swap_trees()

    def resample_manipulator_contacts(self, tree, x):
        # mnp = self.object.sample_contacts(1)
        # ifReturn = True
        # mnp_config = None

        pre_mnp = self.trees[tree].edges[x].manip
        num_manip = self.manipulator.npts
        ifReturn = False
        mnp_config = None
        if pre_mnp is None:
            while not ifReturn:
                mnp = self.object.sample_contacts(num_manip)
                isReachable, mnp_config = self.manipulator.inverse_kinematics(mnp)
                # ifCollide, _ = self.environment.check_collision(self.manipulator, mnp_config)
                ifCollide = self.manipulator.if_collide_w_env(self.environment, mnp_config, x)
                ifReturn = isReachable and (not ifCollide)

        else:
            counter = 0
            max_count = 4
            while counter < max_count and (not ifReturn):
                counter += 1
                mnp = np.array([None]*num_manip)
                # random find contacts that change
                num_manip_left = random.randint(0, num_manip-1)
                manip_left = random.sample(range(num_manip), num_manip_left)

                # check if equilibrium if the selected manip contacts are moved
                if static_equilibrium(x, np.array(pre_mnp)[manip_left], self.trees[tree].edges[x].env, self.world, self.mnp_mu, self.env_mu, self.mnp_fn_max):
                    # randomly sample manipulator contacts
                    mnp[manip_left] = np.array(pre_mnp)[manip_left]
                    for i in range(len(mnp)):
                        if mnp[i] is None:
                            mnp[i] = self.object.sample_contacts(1)[0]
                    # check inverse kinematics
                    isReachable, mnp_config = self.manipulator.inverse_kinematics(mnp)
                    if isReachable:
                        ifCollide = self.manipulator.if_collide_w_env(self.environment, mnp_config, x)
                        if not ifCollide:
                            ifReturn = True
        return ifReturn, mnp, mnp_config

    def inverse_mechanics(self, x, v_star, envs, mnps):

        mnps = [(np.array(m.p), np.array(m.n), m.d) for m in mnps]
        envs = [(np.array(m.p), np.array(m.n), m.d) for m in envs]
        v = inv_planar_2d(np.array(v_star), np.array(x), mnps, envs, self.mnp_mu, self.env_mu)
        # TODO: set friction coefficient here
        return v

    def forward_integration(self, x_near, x_rand, envs, mnps):
        # all collision checking, event detection , timestepping, ...
        h = 0.1
        x = np.array(x_near)
        path = [tuple(x)] # TODO: hack
        g_v = np.identity(3)
        g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]
        v_star = np.dot(g_v.T, np.array(x_rand) - np.array(x))
        if np.linalg.norm(v_star) > h:
            v_star = steer(0, v_star, h)

        while np.linalg.norm(v_star) > 1e-2:

            v = self.inverse_mechanics(x, v_star, envs, mnps)

            if np.linalg.norm(v) < 1e-2:
                break

            # check collision
            x_ = x + np.dot(g_v, v).flatten()
            if_collide, contacts = self.check_collision(x_)
            ifpenetrate = self.check_penetration(contacts)
            if not ifpenetrate:
                # update x if not collide
                x = x_
                path.append(tuple(x))
                envs = contacts
                g_v = np.identity(3)
                g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]
                v_star = np.dot(g_v.T, np.array(x_rand) - x)
                if np.linalg.norm(v_star) > h:
                    v_star = steer(0, v_star, h)
            else:
                # return x backward at collisiton
                depths = np.array([c.d for c in contacts])
                max_i = np.argmax(depths)
                d_max = np.max(depths)
                p_max = contacts[max_i].p
                n_max = contacts[max_i].n
                vs = np.dot(adjointTrans_2d(config2trans(x)), v)
                v_p_max = np.dot(v_hat(vs), np.concatenate([p_max, [1]]))
                k_new = 1 - d_max / abs(np.dot(v_p_max[0:2], n_max))
                if abs(k_new) < 1:
                    v = k_new * v
                    x = x + np.dot(g_v, v)
                    path.append(tuple(x))
                break

        return tuple(x), path

    def connect(self, tree, q_new):
        status = Status.ADVANCED
        while status == Status.ADVANCED:
            x, status = self.extend(tree, q_new)

        return x, status

    def extend(self, tree, x_rand):
        # nearest
        x_near = self.get_nearest(tree, x_rand)
        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip
        # manipulator contacts: sample from the previous one or new mnps
        if random.randint(0,1) == 0 or (self.trees[tree].edges[x_near].manip is None):
            # TODO: get more samples and choose the best with stablity margin
            ifsampled, mnps_new, _ = self.resample_manipulator_contacts(tree, x_near)
            if ifsampled:
                mnps = mnps_new
            elif self.trees[tree].edges[x_near].manip is None:
                return x_near, Status.TRAPPED

        # forward (ITM)
        x_new, path = self.forward_integration(x_near, x_rand, envs, mnps)
        path.reverse()
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 1e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        if status != Status.TRAPPED:
            _, envs = self.check_collision(x_new)
            self.trees[tree].add(x_new, x_near, mnps, envs, path)
            self.samples_taken += 1

        return x_new, status

    def search(self):
        _, envs = self.check_collision(self.x_init)
        self.trees[0].add(self.x_init, None, None, envs)

        x_new, goal_status = self.extend(0, self.x_goal)
        if goal_status == Status.REACHED:
            print(goal_status)
            path = self.reconstruct_path(0, self.x_init, x_new)
            return path

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x_new, status = self.extend(0, x_rand)
            if status != Status.TRAPPED:
                x_new, goal_status = self.extend(0, self.x_goal)
                if goal_status == Status.REACHED:
                    print(goal_status)
                    path = self.reconstruct_path(0, self.x_init, x_new)
                    return path

            print(self.samples_taken)
        path = self.reconstruct_path(0, self.x_init, self.get_nearest(0, self.x_goal))
        return path

    def search_connect(self):
        _, envs = self.check_collision(self.x_init)
        self.trees[0].add(self.x_init, None, None, envs)
        self.add_tree()
        _, envs = self.check_collision(self.x_goal)
        self.trees[1].add(self.x_goal, None, None, envs)

        x_new, goal_status = self.extend(0, self.x_goal)
        if goal_status == Status.REACHED:
            print(goal_status)
            path = self.reconstruct_path(0, self.x_init, x_new)
            return path

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x_new, status = self.extend(0, x_rand)
            if status != Status.TRAPPED:
                _, connect_status = self.connect(1, x_new)
                if connect_status == Status.REACHED:
                    self.unswap()
                    print(connect_status)
                    path0 = self.reconstruct_path(0, self.x_init, self.get_nearest(0,x_new))
                    path1 = self.reconstruct_path(1, self.x_goal, self.get_nearest(1,x_new))
                    path1.reverse()
                    path = path0 + path1
                    return path
            self.samples_taken += 1
            print(self.samples_taken)
        path = self.reconstruct_path(0, self.x_init, self.get_nearest(0, self.x_goal))
        return path

    def search_bidir(self):
        _, envs = self.check_collision(self.x_init)
        self.trees[0].add(self.x_init, None, None, envs)
        # this only applies for quasi-static assumption
        self.add_tree()
        _, envs = self.check_collision(self.x_goal)
        self.trees[1].add(self.x_goal, None, None, envs)
        x_new, goal_status = self.extend(0, self.x_goal)
        if goal_status == Status.REACHED:
            print(goal_status)
            path = self.reconstruct_path(0, self.x_init, x_new)
            return path

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x0, status0 = self.connect(0, x_rand)
            x1, status1 = self.connect(1, x_rand)

            x1_, goal_status1 = self.connect(1, x0)
            x0_, goal_status0 = self.connect(0, x1)

            if goal_status1 == Status.REACHED:
                print(goal_status1)
                path0 = self.reconstruct_path(0, self.x_init, x0)
                path1 = self.reconstruct_path(1, self.x_goal, x1_)
                path1.reverse()
                path = path0 + path1

                return path
            if goal_status0 == Status.REACHED:
                print(goal_status0)
                path0 = self.reconstruct_path(0, self.x_init, x0_)
                path1 = self.reconstruct_path(1, self.x_goal, x1)
                path1.reverse()
                path = path0 + path1
                return path

            print(self.samples_taken)

        path = self.reconstruct_path(0, self.x_init, self.get_nearest(0, self.x_goal))
        return path

class RRTKino_w_modes(RRTKinodynamic):

    def initialize_stability_margin_solver(self):
        self.smsolver = StabilityMarginSolver()

    def add_tree(self):
        """
        Create an empty tree and add to trees
        """
        self.trees.append(RRTTree())

    def reconstruct_path(self, tree, x_init, x_goal):
        """
        Reconstruct path from start to goal
        :param tree: int, tree in which to find path
        :param x_init: tuple, starting vertex
        :param x_goal: tuple, ending vertex
        :return: sequence of vertices from start to goal
        """
        n_nodes = 2
        path = [x_goal]
        current = x_goal
        mnp_path = [None]
        if x_init == x_goal:
            return path
        while not self.trees[tree].edges[current].parent == x_init:
            #path.append(self.trees[tree].E[current])
            n_nodes += 1
            current_path = self.trees[tree].edges[current].path
            path += current_path
            mnp_path += [self.trees[tree].edges[current].manip]*len(current_path)
            current = self.trees[tree].edges[current].parent
        current_path = self.trees[tree].edges[current].path
        path += current_path
        mnp_path += [self.trees[tree].edges[current].manip] * len(current_path)

        path.append(x_init)
        mnp_path.append(None)

        path.reverse()
        mnp_path.reverse()
        print('number of nodes', n_nodes)
        return path, mnp_path

    def nodes_in_ball(self, tree, x):
        nodes = []
        for q in self.trees[tree].nodes:
            d = self.dist(q, x)
            if (d < self.neighbor_radius) and (q != x) and (q not in nodes) and (q in self.trees[tree].edges):
                nodes.append(q)
        return nodes

    def contact_modes(self, x, envs):
        # TODO: number of manipulator contacts should change according to mnp types
        #_, envs = self.check_collision(x)
        modes = get_contact_modes([Contact([],[],None)]*self.manipulator.npts, envs)
        return modes

    def inverse_mechanics(self, x, v_star, envs, mnps, mode):
        if mode is None:
            print('mode cannot be None for RRTKino_w_modes class')
            raise

        # mnps = [(np.array(m.p), np.array(m.n), m.d) for m in mnps]
        # envs = [(np.array(m.p), np.array(m.n), m.d) for m in envs]
        v = qp_inv_mechanics_2d(np.array(v_star), np.array(x), mnps, envs,mode, self.world, self.mnp_mu, self.env_mu, self.mnp_fn_max)
        return v

    def forward_integration(self, x_near, x_rand, envs, mnps, mode):
        # all collision checking, event detection , timestepping, ...
        counter = 0
        h = 0.2
        Status_manipulator_collide = False

        x_rand = np.array(x_rand)
        x = np.array(x_near)
        path = [tuple(x)] # TODO: hack
        g_v = np.identity(3)
        g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]

        v_star = np.dot(g_v.T, x_rand - np.array(x))

        v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
        if np.linalg.norm(v) < 1e-3:
            if v_star[2] > 0:
                x_rand[2] = x_rand[2] - 2*np.pi
            else:
                x_rand[2] = x_rand[2] + 2 * np.pi
            v_star = np.dot(g_v.T, x_rand - np.array(x))
            v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            if np.linalg.norm(v) < 1e-3:
                return tuple(x), path, Status_manipulator_collide

        max_counter = int(np.linalg.norm(v_star) / h)*10
        if np.linalg.norm(v_star) > h:
            v_star = steer(0, v_star, h)

        #TODO: velocity-mode-projection: v_star = v_star*d_proj
        d_proj = velocity_project_direction(v_star, mnps, envs, mode)
        v_star_proj = np.dot(v_star,d_proj)*d_proj
        while np.linalg.norm(v_star) > 1e-2 and counter < max_counter:
            counter += 1
            # v = self.inverse_mechanics(x, v_star, envs, mnps, mode)

            v = self.inverse_mechanics(x, v_star, envs, mnps, mode)
            if np.linalg.norm(v) < 1e-3:
                break

            # check collision
            x_ = x.flatten() + np.dot(g_v, v).flatten()
            _, mnp_config = self.manipulator.inverse_kinematics(mnps) # TODO: need to store mnp_config, no IK everytime
            if self.manipulator.if_collide_w_env(self.environment, mnp_config, x_):
                Status_manipulator_collide = True
                break
            if_collide, contacts = self.check_collision(x_)
            ifpenetrate = self.check_penetration(contacts)
            #ifpenetrate = False
            if not ifpenetrate:
                # update x if not collide

                # check the number of envs contacts
                if len(envs) != len(contacts):
                    if len(contacts) == (sum(np.array(mode) != CONTACT_MODE.LIFT_OFF) - len(mnps)):
                        mode = list(np.array(mode)[np.array(mode) != CONTACT_MODE.LIFT_OFF])
                    else:
                        x = x_
                        path.append(tuple(x))
                        break
                else:
                    is_same_contacts = True
                    for i in range(len(envs)):
                        if not envs[i].is_same(contacts[i]):
                            is_same_contacts = False
                    if not is_same_contacts:
                        break

                x = x_
                path.append(tuple(x))
                envs = contacts
                g_v = np.identity(3)
                g_v[0:2,0:2] = config2trans(x)[0:2, 0:2]
                v_star = np.dot(g_v.T, x_rand - x)
                if np.linalg.norm(v_star) > h:
                    v_star = steer(0, v_star, h)
                v_star_proj = np.dot(v_star,d_proj)*d_proj
            else:
                # return x backward at collisiton
                depths = np.array([c.d for c in contacts])
                max_i = np.argmax(depths)
                d_max = np.max(depths)
                p_max = contacts[max_i].p
                n_max = contacts[max_i].n
                vs = np.dot(adjointTrans_2d(config2trans(x)), v)
                v_p_max = np.dot(v_hat(vs), np.concatenate([p_max, [1]]))
                k_new = 1 - d_max / abs(np.dot(v_p_max[0:2], n_max))
                if abs(k_new) < 1:
                    v = k_new * v
                    x = x + np.dot(g_v, v).flatten()
                    path.append(tuple(x))
                break
        path_ = [path[0]]
        for i in range(len(path)):
            if np.linalg.norm(np.array(path_[-1]) - np.array(path[i])) > h/5:
                path_.append(path[i])

        if path[-1] not in path_:
            path_.append(path[-1])

        if np.linalg.norm(x) > 1000:
            print('something wrong')

        return tuple(x), path_, Status_manipulator_collide

    def extend_w_mode(self, tree, x_near, x_rand, mode):
        # Todo: use stability margin to choose ?

        h = self.step_length
        if np.linalg.norm(np.array(x_near) - np.array(x_rand)) > h:
            x_rand = steer(x_near, x_rand, h)

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip

        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        if random.randint(0,4) == 0 or (self.trees[tree].edges[x_near].manip is None) \
                or self.trees[tree].edges[x_near].manipulator_collide:
            # Todo: sample good manipulator location given conact mode
            ifsampled, mnps_new, _ = self.resample_manipulator_contacts(tree, x_near)
            if ifsampled:
                mnps = mnps_new
            elif self.trees[tree].edges[x_near].manip is None:
                return x_near, Status.TRAPPED, None

        #stability_margin_score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode, 10, self.mnp_fn_max)

        # forward (ITM)
        x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, mnps, mode)
        path.reverse()
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        edge = None
        if status != Status.TRAPPED:
            _, envs = self.check_collision(x_new)
            edge = RRTEdge(x_near, mnps, envs, path, mode)
            edge.manipulator_collide = status_mnp_collide

        return x_new, status, edge

    def extend_w_mode_bias(self, tree, x_near, x_rand, mode):
        # Todo: use stability margin to choose ?

        # h = self.step_length
        # if np.linalg.norm(np.array(x_near) - np.array(x_rand)) > h:
        #     x_rand = steer(x_near, x_rand, h)

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip

        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        if (self.trees[tree].edges[x_near].manip is None) \
                or self.trees[tree].edges[x_near].manipulator_collide:
            # Todo: sample good manipulator location given conact mode
            mnps = self.best_mnp_location(tree, x_near, x_rand, mode)

        if mnps is None:
            return x_near, Status.TRAPPED, None, 0.0

        #stability_margin_score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode, 10, self.mnp_fn_max, self.dist_weight)
        stability_margin_score = 0.0
        # forward (ITM)
        # time1 = time.time()
        x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, mnps, mode)
        # if len(mode) == 5:
        #     x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, [Contact((0.5,0),(-1,0),0)], mode)
        # time2 = time.time()
        # print('forward integration: ',time2-time1, 'dx:', np.array(x_new) - np.array(x_near), 'dx_desire', np.array(x_rand) - np.array(x_near))

        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        edge = None
        if status != Status.TRAPPED:
            path.reverse()
            edge = RRTEdge(x_near, mnps, envs, path, mode)
            edge.manipulator_collide = status_mnp_collide

        return x_new, status, edge, stability_margin_score

    def extend_w_mode_changemnp(self, tree, x_near, x_rand, mode):
        # Todo: use stability margin to choose ?

        # h = self.step_length
        # if np.linalg.norm(np.array(x_near) - np.array(x_rand)) > h:
        #     x_rand = steer(x_near, x_rand, h)

        # collision check, get envs
        envs = self.trees[tree].edges[x_near].env
        # manipulator contacts: sample from the previous one or new mnps
        # Todo: change manipulator location when stability margin is low
        mnps = self.best_mnp_location(tree, x_near, x_rand, mode)

        if mnps is None:
            return x_near, Status.TRAPPED, None, 0.0

        stability_margin_score = 0.0
        #self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode, 10,
        #                                                               self.mnp_fn_max, self.dist_weight)

        # forward (ITM)
        x_new, path, status_mnp_collide = self.forward_integration(x_near, x_rand, envs, mnps, mode)
        path.reverse()
        x_new = tuple(x_new)

        if self.dist(x_near, x_new) < 1e-3:
            status = Status.TRAPPED
        elif self.dist(x_rand, x_new) < 2e-2:
            status = Status.REACHED
        else:
            status = Status.ADVANCED

        # add node and edge
        edge = None
        if status != Status.TRAPPED:
            _, envs = self.check_collision(x_new)
            edge = RRTEdge(x_near, mnps, envs, path, mode)
            edge.manipulator_collide = status_mnp_collide

        return x_new, status, edge, stability_margin_score

    def best_mnp_location(self, tree, x_near, x_rand, mode):

        n_sample = 10

        g_v = np.identity(3)
        g_v[0:2, 0:2] = config2trans(x_near)[0:2, 0:2]
        v_star = np.dot(g_v.T, np.array(x_rand) - np.array(x_near))

        envs = self.trees[tree].edges[x_near].env
        mnps = self.trees[tree].edges[x_near].manip
        mnps_list = []
        score_list = []
        dist_list = []
        if mnps is not None:
            mnps_list.append(mnps)

            score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode,
                                                          10, self.mnp_fn_max, self.dist_weight)
            score_list.append(score)
            v = self.inverse_mechanics(x_near, v_star, envs, mnps, mode)
            dist_list.append(self.dist(v, v_star))

        for i in range(n_sample):
            ifsampled, mnps, _ = self.resample_manipulator_contacts(tree, x_near)
            if ifsampled:
                v = self.inverse_mechanics(x_near, v_star, envs, mnps, mode)
                if np.linalg.norm(v) > 1e-3:
                    mnps_list.append(mnps)

                    score = self.smsolver.compute_stablity_margin(self.env_mu, self.mnp_mu, envs, mnps, mode,
                                                                               10, self.mnp_fn_max, self.dist_weight)
                    score_list.append(score)
                    dist_list.append(self.dist(v, v_star))

        # how to choose the best manipulator location, what's the metric?

        if len(mnps_list) > 0:
            best_score_ind = np.argmax(score_list)
            return mnps_list[best_score_ind]
        else:
            return None

    def cost(self, tree, x , x_edge):
        parent_mode = self.trees[tree].edges[x_edge.parent].mode
        c_env = -2*(len(x_edge.env) + len(self.trees[tree].edges[x_edge.parent].env))
        if x_edge.mode == parent_mode:
            c_inst = -5
        else:
            c_inst = 0.1
        c_dist = self.cost_weight[0]*(self.dist(x, self.x_goal))
        c = self.trees[tree].costs[x_edge.parent] + c_dist + c_env + c_inst

        return c

    def optimal_connect(self, tree, x, x_edge):
        x_star = x
        J_star = self.cost(tree, x, x_edge)
        edge_star = x_edge
        X_near = self.nodes_in_ball(tree, x)
        for x_p in X_near:
            if x_p not in self.trees[tree].enum_modes:
                _, x_p_env = self.check_collision(x_p)
                self.trees[tree].add_mode_enum(x_p, self.contact_modes(x_p, x_p_env))

            for m in self.trees[tree].enum_modes[x_p]:
                x_new, status, edge = self.extend_w_mode(tree, x_p, x, m)
                if status == Status.REACHED:
                    J = self.cost(tree, x_new, edge)
                    if J < J_star:
                        J_star = J
                        edge_star = edge
                        x_star = x_new
                        print('reconnect.')
        self.trees[tree].add(x_star, edge_star)
        self.trees[tree].add_mode_enum(x_star, self.contact_modes(x_star))
        # TODO: add cost
        self.trees[tree].costs[x_star] = J_star
        self.samples_taken += 1
        return x_star

    def rewire(self, tree, x):
        X_near = self.nodes_in_ball(tree, x)
        X_near.remove(self.trees[tree].edges[x].parent)
        modes = self.trees[tree].enum_modes[x]
        for x_child in X_near:
            edge_star = self.trees[tree].edges[x_child]
            J_star = self.trees[tree].costs[x_child]
            x_child_star = x_child
            isrewire = False
            for m in modes:
                x_new, status, edge = self.extend_w_mode(tree, x, x_child, m)
                if status == Status.REACHED:
                    J = self.cost(tree, x_new, edge)
                    if J < J_star:
                        J_star = J
                        edge_star = edge
                        x_child_star = x_new
                        isrewire = True
                        print('rewire.')
            if isrewire:
                del self.trees[tree].edges[x_child]
                del self.trees[tree].costs[x_child]
                del self.trees[tree].enum_modes[x_child]
                self.trees[tree].nodes.remove(x_child)

                self.trees[tree].add(x_child_star, edge_star)
                self.trees[tree].add_mode_enum(x_child_star, self.contact_modes(x_child_star))
                self.trees[tree].costs[x_child_star] = J_star
        return

    # def search(self):
    #     _, envs = self.check_collision(self.x_init)
    #     edge = RRTEdge(None, None, envs, None, None)
    #     self.trees[0].add(self.x_init, edge)
    #     # need to specify mode before extend or connect
    #     init_modes = self.contact_modes(self.x_init)
    #     self.trees[0].add_mode_enum(self.x_init, init_modes)
    #
    #     while self.samples_taken < self.max_samples:
    #         x_rand = self.X.sample_free()
    #         x_near = self.get_nearest(0, x_rand)
    #         near_modes = self.trees[0].enum_modes[x_near]
    #         for m in near_modes:
    #             # TODO: use stability margin to biasly choose contact mode
    #             x_new, status, edge = self.extend_w_mode(0, x_near, x_rand, m)
    #             if status != Status.TRAPPED:
    #                 self.trees[0].add(x_new, edge)
    #                 self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new))
    #                 self.samples_taken += 1
    #
    #
    #         x_near = self.get_nearest(0, self.x_goal)
    #         near_modes = self.trees[0].enum_modes[x_near]
    #         for m in near_modes:
    #             x_new, goal_status,edge = self.extend_w_mode(0, x_near, self.x_goal, m)
    #             if goal_status != Status.TRAPPED:
    #                 self.trees[0].add(x_new, edge)
    #                 self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new))
    #                 self.samples_taken+=1
    #             if goal_status == Status.REACHED:
    #                 print(goal_status)
    #                 path = self.reconstruct_path(0, self.x_init, x_new)
    #                 return path
    #
    #         print(self.samples_taken)
    #     x_nearest = self.get_nearest(0, self.x_goal)
    #     path = self.reconstruct_path(0, self.x_init, x_nearest)
    #     print(x_nearest)
    #     print(self.dist(x_nearest, self.x_goal))
    #     return path

    def search_star(self):
        _, envs = self.check_collision(self.x_init)
        edge = RRTEdge(None, None, envs, None, None)
        self.trees[0].add(self.x_init, edge)
        # need to specify mode before extend or connect
        init_modes = self.contact_modes(self.x_init)
        self.trees[0].add_mode_enum(self.x_init, init_modes)
        self.trees[0].costs[self.x_init] = 0

        while self.samples_taken < self.max_samples:
            x_rand = self.X.sample_free()
            x_near = self.get_nearest(0, x_rand)
            near_modes = self.trees[0].enum_modes[x_near]
            for m in near_modes:
                x_new, status, edge = self.extend_w_mode(0, x_near, x_rand, m)
                if status != Status.TRAPPED:
                    x_new = self.optimal_connect(0, x_new, edge)
                    #self.rewire(0, x_new)

            x_near = self.get_nearest(0, self.x_goal)
            near_modes = self.trees[0].enum_modes[x_near]
            for m in near_modes:
                x_new, goal_status, edge = self.extend_w_mode(0, x_near, self.x_goal, m)
                if goal_status != Status.TRAPPED:
                    x_new = self.optimal_connect(0, x_new, edge)
                    #self.rewire(0, x_new)
                if goal_status == Status.REACHED:
                    print(goal_status)
                    path, mnp_path = self.reconstruct_path(0, self.x_init, x_new)
                    return path, mnp_path

            print(self.samples_taken)

        x_nearest = self.get_nearest(0, self.x_goal)
        path, mnp_path = self.reconstruct_path(0, self.x_init, x_nearest)
        print(x_nearest)
        print(self.dist(x_nearest, self.x_goal))
        return path, mnp_path

    def search(self):
        _, envs = self.check_collision(self.x_init)
        edge = RRTEdge(None, None, envs, None, None)
        self.trees[0].add(self.x_init, edge)
        init_modes = self.contact_modes(self.x_init, envs)
        self.trees[0].add_mode_enum(self.x_init, init_modes)

        while self.samples_taken < self.max_samples:
            x_nearest, d_nearest = self.get_goal_nearest(0)
            if d_nearest < 0.02:
                print('nearest state: ', x_nearest, ', dist: ', d_nearest)
                print('GOAL REACHED.')
                path, mnp_path = self.reconstruct_path(0, self.x_init, x_nearest)
                return path, mnp_path

            if random.randint(0,1) == 0:
                x_rand = self.X.sample_free()
            else:
                x_rand = self.x_goal

            x_near = self.get_nearest(0, x_rand)
            near_modes = self.trees[0].enum_modes[x_near]
            for m in near_modes:
                # TODO: use stability margin to biasly choose contact mode
                x_new, status, edge = self.extend_w_mode(0, x_near, x_rand, m)
                if status != Status.TRAPPED:
                    self.trees[0].add(x_new, edge)
                    self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new, edge.env))
                    self.samples_taken += 1
                    print('sample ', self.samples_taken)

        x_nearest = self.get_nearest(0, self.x_goal)
        path, mnp_path = self.reconstruct_path(0, self.x_init, x_nearest)
        print('nearest state: ', x_nearest, ', dist: ', self.dist(x_nearest, self.x_goal))
        return path, mnp_path

    def search_bias(self):
        _, envs = self.check_collision(self.x_init)
        edge = RRTEdge(None, None, envs, None, None)
        self.trees[0].add(self.x_init, edge)
        init_modes = self.contact_modes(self.x_init, envs)
        self.trees[0].add_mode_enum(self.x_init, init_modes)

        # x_near = (0.6829629749145216, 1.9999999392151775, -1.5707962674487148)
        # _, envs = self.check_collision(x_near)
        # mnps = [Contact((-0.45, 0.5), (0, -1), 0), Contact((0.3,-0.5), (0,1), 0)]
        # mode = [CONTACT_MODE.FOLLOWING, CONTACT_MODE.FOLLOWING, CONTACT_MODE.LIFT_OFF, CONTACT_MODE.STICKING]
        # self.forward_integration(x_near, self.x_goal, envs, mnps, mode)


        thr_stability = 0.1
        while self.samples_taken < self.max_samples:
            x_nearest, d_nearest = self.get_goal_nearest(0)
            if d_nearest < 0.04:
                print('nearest state: ', x_nearest, ', dist: ', d_nearest)
                print('GOAL REACHED.')
                path, mnp_path = self.reconstruct_path(0, self.x_init, x_nearest)
                return path, mnp_path

            if random.randint(0,3) == 0:
                #x_rand = self.x_goal
                x_rand = self.X.sample_free()
                x_near = self.get_nearest(0, x_rand)
            else:
                x_rand = self.x_goal
                x_near = self.get_unexpand_nearest(0)
                self.trees[0].goal_expand[x_near] = True

            trapped_status_list = []
            stability_score_list = []
            _, envs = self.check_collision(x_near)
            self.trees[0].edges[x_near].env = envs
            near_modes = self.contact_modes(x_near, envs)
            for m in near_modes:
                # TODO: use stability margin to biasly choose contact mode
                x_new, status, edge, score = self.extend_w_mode_bias(0, x_near, x_rand, m)
                stability_score_list.append(score)
                trapped_status_list.append(status == Status.TRAPPED)
                if status != Status.TRAPPED and self.dist(x_new, self.get_nearest(0,x_new)) > 1e-3:
                    self.trees[0].add(x_new, edge)
                    self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new, edge.env))
                    self.samples_taken += 1
                    print('sample ', self.samples_taken,', x: ', x_new)
                    # add nodes on the path
                    # d_i = int(len(edge.path)/3)+1
                    # i = d_i
                    # while i < len(edge.path):
                    #     x_new = edge.path[i]
                    #     _, envs = self.check_collision(x_new)
                    #     edge_ = RRTEdge(x_near, edge.manip, envs, edge.path[0:i+1], edge.mode)
                    #     self.trees[0].add(x_new, edge_)
                    #     self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new, edge_.env))
                    #     i += d_i

                else:
                    # only if project velocity direction is positive
                    x_new, status, edge, score = self.extend_w_mode_changemnp(0, x_near, x_rand, m)
                    if status != Status.TRAPPED and self.dist(x_new, self.get_nearest(0,x_new)) > 1e-3:
                        self.trees[0].add(x_new, edge)
                        self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new, edge.env))
                        self.samples_taken += 1
                        print('sample ', self.samples_taken, ', x: ',  x_new, 'change mnp')
                        # add nodes on the path
                        # d_i = int(len(edge.path) / 3) + 1
                        # i = d_i
                        # while i < len(edge.path):
                        #     x_new = edge.path[i]
                        #     _, envs = self.check_collision(x_new)
                        #     edge_ = RRTEdge(x_near, edge.manip, envs, edge.path[0:i], edge.mode)
                        #     self.trees[0].add(x_new, edge_)
                        #     self.trees[0].add_mode_enum(x_new, self.contact_modes(x_new,edge_.env))
                        #     i+=d_i

        x_nearest, d_nearest = self.get_goal_nearest(0)
        print('nearest state: ', x_nearest, ', dist: ', d_nearest)
        path, mnp_path = self.reconstruct_path(0, self.x_init, x_nearest)

        return path, mnp_path






