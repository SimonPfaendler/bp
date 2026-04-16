import numpy as np

# Returns Velocity to turn Robot to given Point
def turn_to_point(robot, point):
    target_angle_deg = np.degrees(np.arctan2(point[1] - robot.y, point[0] - robot.x))
    robot_deg = robot.theta
    angle_diff = (target_angle_deg - robot_deg + 180) % 360 - 180
    #angular_velocity = np.clip(angle_diff / 180.0, -1.0, 1.0)
    angular_velocity = np.tanh(angle_diff / 45.0)

    return angular_velocity


def turn_to_object(robot, object):
    return turn_to_point(robot, np.array([object.x, object.y]))


def turn_away_from_object(robot, object):
    target_angle_deg = np.degrees(np.arctan2(object.y - robot.y, object.x - robot.x))
    target_angle_deg = (target_angle_deg + 180) % 360
    robot_deg = robot.theta
    angle_diff = (target_angle_deg - robot_deg + 180) % 360 - 180
    angular_velocity = np.tanh(angle_diff / 45.0)

    return angular_velocity


def move_to_point(robot, point, speed=1.0):
    robot_pos = np.array([robot.x, robot.y])
    direction = point - robot_pos
    distance = np.linalg.norm(direction)

    
    if distance < 0.09:
        return 0.0, 0.0

    BRAKE_START = 0.4
    if distance < BRAKE_START:
        speed_scale = speed * (distance - 0.09) / (BRAKE_START - 0.09)
    else:
        speed_scale = speed

    v_x, v_y = (direction / distance) * speed_scale
    return float(v_x), float(v_y)


def move_to_ball(robot, ball, speed=1.0):
    if robot.infrared is True:
        return np.array([0.0, 0.0, 0.0, 0.0, 1.0])
        
    robot_pos = np.array([robot.x, robot.y])
    ball_pos = np.array([ball.x, ball.y])

    
    target_angle_deg = np.degrees(np.arctan2(ball_pos[1] - robot.y, ball.x - robot.x))
    angle_diff = (target_angle_deg - robot.theta + 180) % 360 - 180
    angular_velocity = np.clip(angle_diff / 15.0, -1.0, 1.0)

    direction = ball_pos - robot_pos
    distance = np.linalg.norm(direction)

    
    if distance < 0.13: # robot_radius + ball_radius
        speed_scale = speed * 0.2
    else:
        speed_scale = speed

    v_x, v_y = (direction / distance) * speed_scale

    return np.array([v_x, v_y, angular_velocity, 0.0, 0.0])


# Shoot Ball at Goal
def shoot_at_goal_center(env, robot, team_color):
    goal_x = -env.field.length / 2.0 if team_color == "yellow" else env.field.length / 2.0
    goal_y = 0.0
    target_point = np.array([goal_x, goal_y])
    
    
    v_theta = turn_to_point(robot, target_point)
    if abs(v_theta) < 0.1:

        return np.array([0.0, 0.0, 0.0, 6.0, 0.0])
    else:
        return np.array([0.0, 0.0, v_theta, 0.0, 1.0])


def shoot_at_point(robot, target_point):
    v_theta = turn_to_point(robot, target_point)

    if abs(v_theta) < 0.15:
        
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0])
    else:
        return np.array([0.0, 0.0, v_theta, 0.0, 1.0])


def dribble_to_point(robot, point, speed=0.8):

    v_x, v_y = move_to_point(robot, point, speed=speed)
    v_theta = turn_to_point(robot, point)
    return np.array([v_x, v_y, v_theta, 0.0, 1.0])
