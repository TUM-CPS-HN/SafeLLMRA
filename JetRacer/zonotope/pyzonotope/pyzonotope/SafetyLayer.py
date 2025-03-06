import numpy as np
from multiprocessing import Pool
from pyzonotope.helper import check_zono_intersection
import math
from pyzonotope.Zonotope import Zonotope
from pyzonotope.NLReachability import NLReachability



class Pose:
    def __init__(self, *args):
        if(len(args) == 2):
            self.x = args[0] 
            self.y = args[1] 
        elif(len(args) == 1):
            self.x = args[0][0]
            self.y = args[0][1]
        
    def __sub__(self, *factor):
        result = Pose(self.x, self.y)
        if(len(factor) == 1):
            if(isinstance(factor[0], Pose)):
                factor = factor[0]
                result.x -= factor.x
                result.y -= factor.y                
            else:
                result.x -= factor[0]
                result.y -= factor[0]
        elif(len(factor) == 2):
            result.x -= factor[0]
            result.y -= factor[1]
            
        return result
    
    def __add__(self, *factor):
        result = Pose(self.x, self.y)
        if(len(factor) == 1):
            if(isinstance(factor[0], Pose)):
                factor = factor[0]
                result.x += factor.x
                result.y += factor.y                
            else:
                result.x += factor[0]
                result.y += factor[0]
        elif(len(factor) == 2):
            result.x += factor[0]
            result.y += factor[1]
            
        return result    
    def __mul__(self, factor):
        result = Pose(factor * self.x, factor * self.y)
        return result
    
    def __truediv__(self, factor):
        result = Pose(self.x / factor, self.y / factor)
        return result
        
    
    def as_tuple(self):
        return (self.x, self.y)
    
    def __str__(self):
        result = "X: {}, Y: {}".format(self.x, self.y)
        return result


class SafetyLayer():
    def __init__(self, env = "tb"):
        self.env = env
        self.pool = Pool()

        
        #self.nonlinear_reachability = NLReachability("/home/ali/Downloads/Data-Driven-Reachability-Analysis-main/examples-basic")
        self.nonlinear_reachability = NLReachability("/home/jetson/catkin_ws/src/py_zonotope-main/example")
        self.old_plan = []
        for i in range(3):
            self.old_plan.append(np.array([0.5, 1]))
        self.old_plan = np.array(self.old_plan)
        self.pool = Pool()
        
        self.fail_safe = []
        for i in range(1):
            self.fail_safe.append([0., 0.25])
        self.fail_safe = np.array(self.fail_safe)








    def get_rays_poses2d(self, readings, yaw,current_position):


        if len(readings) == 1:
            angles = [0]
        else:
            angles = [-np.pi/2 + i * (np.pi / (len(readings) - 1)) for i in range(len(readings))]

        # Compute original poses (before rotation)
        poses = [np.array([np.cos(angle), np.sin(angle)]) for angle in angles]

        # Scale unit vectors by readings
        rays_poses = [poses[i] * readings[i] for i in range(len(readings))]

        # Create the rotation matrix
        rotation_matrix = np.array([[np.cos(yaw), -np.sin(yaw)], 
                                    [np.sin(yaw),  np.cos(yaw)]]).T

        # Apply rotation to each ray (Fixing shape issue)
        rotated_rays = [rotation_matrix @ ray.reshape(2, 1)+current_position for ray in rays_poses]
        rotated_rays = [ray.flatten() for ray in rotated_rays]  # Convert back to 1D

        return rotated_rays



    """def construct_objects(self, readings, _range = None):
        objects = []

        if(self.env == "tb"):
            print("CONSTRUCTING OBSTACLES") 
            active_rays = np.argwhere(readings <= 1.0).flatten()
            if(_range is None):
                _range = [-90, 90]


            i = 0
            while i < len(active_rays) - 2:
                if(active_rays[i] + 1 == active_rays[i + 1] and active_rays[i] + 2 == active_rays[i + 2]):
                    objects.append([active_rays[i], active_rays[i + 1], active_rays[i + 2]])
                    i += 3

                elif(active_rays[i] + 1 == active_rays[i + 1]):
                    objects.append([active_rays[i], active_rays[i + 1]])
                    i += 2

                else:
                    i += 1
                    continue

            remaining_active_rays = len(active_rays) - i

            if(remaining_active_rays == 2):
                objects.append([active_rays[i], active_rays[i + 1]])
            elif(remaining_active_rays == 1):
                if(active_rays[i] - 1 == active_rays[i - 1]):
                    objects.append([active_rays[i - 1], active_rays[i]])

        else:
            pass

        return objects"""
        
    def construct_objects(self, readings, _range=None, max_gap=1, min_cluster_size=2, max_distance_diff=0.3):
      """
      Improved object clustering from lidar readings.
      
      Parameters:
      - readings: array of lidar distance readings
      - _range: angle range to consider [min_angle, max_angle]
      - max_gap: maximum gap between consecutive rays to be considered part of the same object
      - min_cluster_size: minimum number of points to form a valid cluster
      - max_distance_diff: maximum difference in distance between consecutive points in a cluster
      
      Returns:
      - List of objects, where each object is a list of ray indices
      """
      objects = []
      
      if self.env == "tb":
          # Find active rays (those that detect something within range)
          active_rays = np.argwhere(readings <= 2.0).flatten()
          
          if _range is None:
              _range = [-90, 90]
          
          # Skip if no active rays
          if len(active_rays) == 0:
              return objects
              
          # Start with the first ray as a new cluster
          current_cluster = [active_rays[0]]
          prev_ray = active_rays[0]
          prev_distance = readings[prev_ray]
          
          # Process remaining rays
          for i in range(1, len(active_rays)):
              current_ray = active_rays[i]
              current_distance = readings[current_ray]
              
              # Check if this ray should be part of the current cluster
              # Criteria: 1) Ray index gap is small, 2) Distance difference is not too large
              if (current_ray - prev_ray <= max_gap and 
                  abs(current_distance - prev_distance) <= max_distance_diff):
                  # Add to current cluster
                  current_cluster.append(current_ray)
              else:
                  # Check if current cluster is large enough to be valid
                  if len(current_cluster) >= min_cluster_size:
                      objects.append(current_cluster)
                  # Start a new cluster
                  current_cluster = [current_ray]
              
              prev_ray = current_ray
              prev_distance = current_distance
          
          # Don't forget the last cluster
          if len(current_cluster) >= min_cluster_size:
              objects.append(current_cluster)
              
      else:
          pass
      
      return objects
     



    
    def create_zonotope2d(self, indices, readings,yaw,current_position):
        zonotopes = []
        poses = self.get_rays_poses2d(readings,yaw,current_position)
        for idx in indices:
            first, last = poses[idx[0]], poses[idx[-1]]
            first_gen = np.array([last[0] - first[0], last[1] - first[1]])
            second_gen = np.array(poses[idx[0]])
            third_gen = np.array(poses[idx[-1]])


            first_gen_center = (poses[idx[0]] + poses[idx[-1]]) / 2
            perpendicular = np.array([-first_gen[1] / first_gen[0], 1])
            perpendicular = math.sqrt(3) * (perpendicular / np.linalg.norm(perpendicular))

            center = np.array((0, 0)).reshape((-1, 1))

            first_gen = first_gen / 2
            second_gen = np.linalg.norm(first_gen) * (second_gen / np.linalg.norm(second_gen))
            third_gen = np.linalg.norm(first_gen) * (third_gen / np.linalg.norm(third_gen))

            generators = np.hstack([first_gen.reshape((-1, 1)), second_gen.reshape((-1, 1)), \
                                                   third_gen.reshape((-1, 1))])


            centered_zonotope = Zonotope(center, generators)
            center_shift = first_gen_center
            vertices = centered_zonotope.polygon()
            for i in range(vertices.shape[1] - 1):
                for j in range(1, vertices.shape[1]):
                    line = vertices[:, i] - vertices[:, j]
                    line_center = (vertices[:, i] + vertices[:, j]) / 2
                 
                    if(np.abs(np.arctan2(line[1], line[0]) - np.arctan2(first_gen[1], first_gen[0])) < 0.001 and # Same Slope
                       (np.linalg.norm(line) / 2) - np.linalg.norm(first_gen) < 0.001 and
                       line_center @ first_gen_center < 0):
                        center_shift -= line_center
                        zonotopes.append(Zonotope(center_shift.reshape((-1, 1)), generators))
                   
        return zonotopes    
    




    
    def enforce_safety(self, reachability_state, plan, readings):
        obstacles_indices = self.construct_objects(1.3 * readings)
        obstacles = self.create_zonotope2d(obstacles_indices, readings)
        if(len(obstacles) > 0):
             

                new_states  = self.nonlinear_reachability.Linear_Reachability(reachability_state,plan)
                
                pose_states = [Zonotope(i.Z[:plan.shape[1], :plan.shape[1] + 1]) for i in new_states[1:]]

                new_states_rep = np.array(pose_states).reshape((-1, 1))
            
                new_states_rep = np.hstack([new_states_rep] * len(obstacles)).flatten()
                zono_obs_pair = zip(new_states_rep, obstacles * len(new_states[1:]))

                ret = self.pool.map(check_zono_intersection, zono_obs_pair)


                ret = np.array(ret, dtype = object)
                
              
        else:
            ret = False        
        return ret,obstacles,new_states
    




    
    def enforce_safety_nonlinear(self, reachability_state, plan, readings,yaw2,current_position):
        obstacles_indices = self.construct_objects( readings)
        obstacles = self.create_zonotope2d(obstacles_indices, readings, yaw2,current_position)

        if(len(obstacles) > 0):
            for j in range(3):
                new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                
                pose_states = []

                # Iterate over all elements in new_states, except the first one
                for i in new_states[0:]:
                    # Extract a specific part of i.Z
                    center_2D = np.array(i.center())
                    generators_2D = np.array(i.generators())
                    new_center = center_2D[:2].reshape(2, 1)  # First two rows of the center
                    new_generators = generators_2D[:2, :]  # First two rows of the generators
                    reachability_state_2D = Zonotope(new_center, new_generators)
                    
                    # Append the Zonotope object to pose_states list
                    pose_states.append(reachability_state_2D)



                new_states_rep = np.array(pose_states).reshape((-1, 1))
            
                new_states_rep = np.hstack([new_states_rep] * len(obstacles)).flatten()
                zono_obs_pair = zip(new_states_rep, obstacles * len(new_states[1:]))

                ret = self.pool.map(check_zono_intersection, zono_obs_pair)

                ret = np.array(ret, dtype = object)
                ret1 = np.array(ret, dtype = object)

                if(any(ret[:, 0]) and j<=2):
                   
                    
                        ret = ret.reshape((len(new_states[1:]), len(obstacles), 2))
                        plan_updates = np.array([np.zeros_like(np.array([0, 0]))] * len(new_states[1:])).astype(np.float32)
                        upstream_gradient = np.ones_like(np.array([0, 0])).astype(np.float32).reshape((1, -1))

                        # Loop backwards to do gradient updates for the plan
                        for i in range(len(ret) - 1, -1, -1):
                            colliding_obstacles = ret[i][np.where(ret[i][:, 0])]
                            if(len(colliding_obstacles) != 0):
                                avg_gradient = np.mean(colliding_obstacles[:, 1]) 
                            else:
                                avg_gradient = np.zeros_like((np.array([0, 0]))).reshape((-1, 1))

                            if(i == len(ret) - 1):
                                plan_updates[i] = (avg_gradient.T @ derivatives[i][1]).flatten()
                            else:
                                plan_updates[i] = ((avg_gradient.T  + upstream_gradient) @ \
                                                (derivatives[i][1])).flatten()
                                

                            upstream_gradient = upstream_gradient @ derivatives[i][0]


                        """plan[:, :2] -= (10 * plan_updates)
                        plan[0, 0] = np.clip(plan[0, 0], 0, 0.1)
                        plan[1:, 0] = np.clip(plan[1:, 0], 0, 0.5)
                        plan[:, 1] = np.clip(plan[:, 1], -0.5, 0.5)"""
                        plan[:, :2] -= (2 * plan_updates)
                        plan[np.where(plan[:, 0] > 0.1), 0] = 0.1
                        plan[np.where(plan[:, 0] < 0), 0]    = 0
                        plan[np.where(plan[:, 1] > 0.5), 1]  = 0.5
                        plan[np.where(plan[:, 1] < -0.5), 1] = -0.5

                else:
                        break
                        
            if(j == 0):
                print("plan is safe :)")

                return new_states,obstacles,True, plan, True
            elif(np.any(ret1[:, 0])):
                print("Update failed")
                #new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,self.old_plan)
                return new_states,obstacles,False, self.old_plan,False
            else:
                print("Update Succedded")
                #new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                
                return new_states,obstacles,True, plan,False
                      
        else:
            ret = np.array([False, 0], dtype=object)
            obstacles = []
            new_states = []
            Flag = True
        return new_states,obstacles,Flag, plan,True
    


    
    def enforce_safety_nonlinear_test(self, reachability_state, plan, readings, yaw2,current_position):
        obstacles_indices = self.construct_objects(readings)
        obstacles = self.create_zonotope2d(obstacles_indices, readings, yaw2,current_position)
        new_states1,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)

        if(len(obstacles) > 0):
            for j in range(3):
                new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                
                pose_states = []

                # Iterate over all elements in new_states, except the first one
                for i in new_states[1:]:
                    # Extract a specific part of i.Z
                    center_2D = np.array(i.center())
                    generators_2D = np.array(i.generators())
                    new_center = center_2D[:2].reshape(2, 1)  # First two rows of the center
                    new_generators = generators_2D[:2, :]  # First two rows of the generators
                    reachability_state_2D = Zonotope(new_center, new_generators)
                    
                    # Append the Zonotope object to pose_states list
                    pose_states.append(reachability_state_2D)



                new_states_rep = np.array(pose_states).reshape((-1, 1))
            
                new_states_rep = np.hstack([new_states_rep] * len(obstacles)).flatten()
                zono_obs_pair = zip(new_states_rep, obstacles * len(new_states[1:]))

                ret = self.pool.map(check_zono_intersection, zono_obs_pair)

                ret = np.array(ret, dtype = object)

                if(any(ret[:, 0]) and j<=2):
                   
                    
                        ret = ret.reshape((len(new_states[1:]), len(obstacles), 2))
                        plan_updates = np.array([np.zeros_like(np.array([0, 0]))] * len(new_states[1:])).astype(np.float32)
                        upstream_gradient = np.ones_like(np.array([0, 0])).astype(np.float32).reshape((1, -1))

                        # Loop backwards to do gradient updates for the plan
                        for i in range(len(ret) - 1, -1, -1):
                            colliding_obstacles = ret[i][np.where(ret[i][:, 0])]
                            if(len(colliding_obstacles) != 0):
                                avg_gradient = np.mean(colliding_obstacles[:, 1]) 
                            else:
                                avg_gradient = np.zeros_like((np.array([0, 0]))).reshape((-1, 1))

                            if(i == len(ret) - 1):
                                plan_updates[i] = (avg_gradient.T @ derivatives[i][1]).flatten()
                            else:
                                plan_updates[i] = ((avg_gradient.T  + upstream_gradient) @ \
                                                (derivatives[i][1])).flatten()
                                

                            upstream_gradient = upstream_gradient @ derivatives[i][0]


                        plan[:, :2] -= (2 * plan_updates)
                        plan[np.where(plan[:, 0] > 0.1), 0] = 0.1
                        plan[np.where(plan[:, 0] < 0), 0]    = 0
                        plan[np.where(plan[:, 1] > 0.5), 1]  = 0.5
                        plan[np.where(plan[:, 1] < -0.5), 1] = -0.5
                else:
                        break
                        
            if(j == 0):
                print("plan is safe :)")

                return new_states1,obstacles,True, plan
            elif(np.any(ret[:, 1])):
                print("Update failed")
                new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                print(plan)
                return new_states,obstacles,False, plan
            else:
                print("Update Succedded")
                new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                print(plan)
                
                return new_states,obstacles,True, plan
                      
        else:
            ret = np.array([False, 0], dtype=object)
            obstacles = []
            new_states = []
            Flag = True
        return new_states,obstacles,Flag, plan
    


    def enforce_unsafe_nonlinear(self, reachability_state, plan, readings, yaw2,current_position):
        obstacles_indices = self.construct_objects(readings)
        obstacles = self.create_zonotope2d(obstacles_indices, readings, yaw2,current_position)

        new_states,derivatives = self.nonlinear_reachability.run_reachability(reachability_state,plan)
                

        return new_states,obstacles, plan
    

