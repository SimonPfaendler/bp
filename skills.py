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

    if distance < 1.0:
        speed_scale = min(speed, float(distance))
    else:
        speed_scale = speed

    v_x, v_y = direction / distance * speed_scale

    #return np.array([v_x, v_y, turn_to_point(robot, point), 0, 0])
    return v_x, v_y


def move_to_ball(robot, ball, speed=1.0):
    robot_radius = 0.09
    ball_radius = 0.0215

    # Distance to the ball at which the robot starts to slow down, to avoid circling
    angle_adjust_dist = 1.6

    if robot.infrared is True:
        return np.array([0, 0, 0, 0, 1.0])
    else:
        robot_pos = np.array([robot.x, robot.y])
        ball_pos = np.array([ball.x, ball.y])

        target_angle_deg = np.degrees(np.arctan2(ball_pos[1] - robot.y, ball.x - robot.x))
        robot_deg = robot.theta
        angle_diff = (target_angle_deg - robot_deg + 180) % 360 - 180
        angular_velocity = np.tanh(angle_diff / 32.0)

        direction = ball_pos - robot_pos
        distance = np.linalg.norm(direction)

        if distance < robot_radius + ball_radius + angle_adjust_dist:
            speed_scale = speed * max(0.1, 1.0 - abs(angle_diff) / 16)
        else:
            speed_scale = speed

        v_x, v_y = direction / distance * speed_scale

        return np.array([v_x, v_y, angular_velocity, 0, 0.0])

#Shoot Ball at Goal
def shoot_goal(env, robot, robot_color):
    robot_pos = np.array([robot.x, robot.y])

    best_target, _ = goal_trajectory_target_and_dist(env, robot, robot_color)

    theta_rad = np.deg2rad(robot.theta)
    robot_forward = np.array([np.cos(theta_rad), np.sin(theta_rad)])

    robot_to_goal = best_target - robot_pos

    t = robot_to_goal @ robot_forward
    t = np.clip(t, 0, None)

    closest_point = robot_pos + t * robot_forward
    dist = np.linalg.norm(best_target - closest_point)

    if dist <= env.field.ball_radius:
        return np.array([0, 0, 0, 1.0, 0])
    else:
        return np.array([0, 0, turn_to_point(robot, best_target), 0, 1.0])


def shoot_at_goal_center(env, robot, team_color):
    goal_x = -env.field.length / 2.0 if team_color == "yellow" else env.field.length / 2.0
    goal_y = 0.0
    target_point = np.array([goal_x, goal_y])
    
    
    v_theta = turn_to_point(robot, target_point)
    if abs(v_theta) < 0.1:

        return np.array([0.0, 0.0, 0.0, 4.0, 0.0])
    else:
        return np.array([0.0, 0.0, v_theta, 0.0, 1.0])

def dribble_to_point(robot, point, speed=0.7):

    v_x, v_y = move_to_point(robot, point, speed=speed)
    v_theta = turn_to_point(robot, point)
    
    
    return np.array([v_x, v_y, v_theta, 0.0, 1.0])
