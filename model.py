import math


"""
ContinuumRobotModel: Class for the continuum robot model.

This file implements the mathematical model of a continuum robot with three cables.  
Main function:
- calculate_position(d1, d2, d3): Calculates the end-effector position (X, Y, Z) based on cable retractions.
Key parameters:
- lq: Robot length without cable retractions (initial length).
- r: Radius of the circle that describes the geometry of the cables.
Features:
- Calculates curvature (kappa), inclination angle (theta), and azimuthal angle (phi).
- Accounts for physical constraints, such as when all cables are identical, the robot remains in its initial state.
"""



class ContinuumRobotModel:
    def __init__(self, lq=110, r=5):
        self.lq = lq  
        self.r = r  

    def calculate_position(self, d1, d2, d3):

        l1 = self.lq + d1  
        l2 = self.lq + d2  
        l3 = self.lq + d3  

        if (d1 == d2 == d3) or (l1 == l2 == l3):
            return 0, 0, self.lq

        
        kappa = (2 * math.sqrt(l1**2 + l2**2 + l3**2 - l1 * l2 - l1 * l3 - l2 * l3)) / \
                (self.r * (l1 + l2 + l3))

       
        theta = (2 * math.sqrt(l1**2 + l2**2 + l3**2 - l1 * l2 - l1 * l3 - l2 * l3)) / \
                (self.r * 3)

       
        phi = math.atan2(math.sqrt(3) * (l2 + l3 - 2 * l1), (3 * (l2 - l3)))

        
        if -math.pi / 2 <= theta <= math.pi / 2 and kappa > 0:
            x = (2 / kappa) * (math.sin(kappa * self.lq / 2)**2) * math.cos(phi)
            y = (2 / kappa) * (math.sin(kappa * self.lq / 2)**2) * math.sin(phi)
            z = (1 / kappa) * math.sin(kappa * self.lq)
            return x, y, z
        else:
            
            return 0, 0, self.lq
        

