%% Prepare workspace
clc;
clear;
close all;

%% Init robotarium
initial_positions = [-1;-0.2;-pi];
robotarium = Robotarium('NumberOfRobots', 1, 'ShowFigure', true, "InitialConditions", initial_positions);
axis on
boundary_min = [robotarium.boundaries(1), robotarium.boundaries(3)];
boundary_max = [robotarium.boundaries(2), robotarium.boundaries(4)];

%% Define parameters
% Define dangerous circles
% Centers [n x 2]
unsafe_circle_centers = [0, 0;
                         1, 1];
% Radii [n]
unsafe_circle_radii = [0.25;
                       0.15];

real_time = true;

waypoint_advance_distance = 0.2;

%% Barrier function
alpha_lse = 10; % log-sum-exp alpha
alpha_cbf = 2; % cbf alpha

% x [3 x 1]
h = @(x) - 1/alpha_lse * log(sum(exp(-alpha_lse*(vecnorm(x(1:2) - unsafe_circle_centers').^2 - unsafe_circle_radii'.^2))));
dhdx = @(x) [sum(exp(-alpha_lse*(vecnorm(x(1:2) - unsafe_circle_centers').^2 - unsafe_circle_radii'.^2))' .* (2*x(1:2)' - 2*unsafe_circle_centers), 1) / ...
            sum(exp(-alpha_lse*(vecnorm(x(1:2) - unsafe_circle_centers').^2 - unsafe_circle_radii'.^2))), 0];

%% Unicycle Dynamics
l = 0.05;
g = @(x) [cos(x(3)), 0;
          sin(x(3)), 0;
                  0, 1];
pxdot_to_u = @(x) [     cos(x(3)),     sin(x(3));
                   -1/l*sin(x(3)), 1/l*cos(x(3))];

%% Plot barrier function heatmap
x_y_grid_ratio = (boundary_max(1)-boundary_min(1))/(boundary_max(2)-boundary_min(2));
num_pixels_x = 1000;
num_pixels_y = round(num_pixels_x/x_y_grid_ratio);

% Calculate heatmap
heatmap_x_bounds = boundary_min(1) + (boundary_max(1)-boundary_min(1)) * (1/num_pixels_x * ([1,num_pixels_x]-0.5));
heatmap_y_bounds = boundary_min(2) + (boundary_max(2)-boundary_min(2)) * (1/num_pixels_y * ([1,num_pixels_y]-0.5));
heatmap_c = zeros(num_pixels_y, num_pixels_x);
for index_y = 1:num_pixels_y
    for index_x = 1:num_pixels_x
        x = boundary_min(1) + (boundary_max(1)-boundary_min(1)) * (1/num_pixels_x * (index_x-0.5));
        y = boundary_min(2) + (boundary_max(2)-boundary_min(2)) * (1/num_pixels_y * (index_y-0.5));
        heatmap_c(index_y, index_x) = h([x;y]);
    end
end

% Create colormap
max_c = max(heatmap_c, [], "all");
min_c = min(heatmap_c, [], "all");
trans_c = 0.2*min(abs(max_c),abs(min_c)); % Transition c value
color_blue      = [   0,   0,   1];
color_lightblue = [ 0.6, 0.6,   1];
color_white     = [   1,   1,   1];
color_lightred  = [   1, 0.6, 0.6];
color_red       = [   1,   0,   0];
custom_map = interp1([max_c; trans_c; 0; -trans_c; min_c], [color_blue; color_lightblue; color_white; color_lightred; color_red], linspace(min_c, max_c, 2000));

imagesc(heatmap_x_bounds, heatmap_y_bounds, heatmap_c, "AlphaData", 0.8);
colormap(custom_map);
colorbar;
clim([min_c, max_c]);

%% Demo
waypoint = [1; 0];
starX = 0.1 * [  0; -0.1123; -0.4755; -0.1816; -0.2939;       0;  0.2939;  0.1816; 0.4755; 0.1123;   0];
starY = 0.1 * [0.5;  0.1545;  0.1545; -0.0590; -0.4045; -0.1910; -0.4045; -0.0590; 0.1545; 0.1545; 0.5];

star_patch = patch("XData", waypoint(1) + starX, ...
    "YData", waypoint(2) + starY, ...
    "FaceColor", [1, 0.7216, 0], "EdgeColor", [0.5, 0.3529, 0], "LineWidth", 2);

sound_laughter = load("laughter.mat").y;
sound_handel = load("handel.mat").y;

while(true)
    rx = robotarium.get_poses; % [3 x N]
    rpos = rx(1:2);
    rtheta = rx(3);

    % h_current = h(rx);

    % Generate desired command
    max_mag_pdot = 0.05;

    pxdot = 1*(waypoint - rpos);
    pxdot = pxdot * min(1, max_mag_pdot/norm(pxdot));

    u_desired = pxdot_to_u(rx) * pxdot;

    % Calculate cbf stuff
    cx = unsafe_circle_centers(1,1);
    cy = unsafe_circle_centers(1,2);
    r = unsafe_circle_radii(1);

    h_current = (rx(1) + l*cos(rtheta) - cx)^2 + (rx(2) + l*sin(rtheta) - cy)^2 - r^2;

    Lgh = [2*(rx(1)+l*cos(rtheta)-cx)*cos(rtheta) + 2*(rx(2)+l*sin(rtheta)-cy)*sin(rtheta), ...
        -2*l*(rx(1)+l*cos(rtheta)-cx)*sin(rtheta)+2*l*(rx(2)+l*sin(rtheta)-cy)*cos(rtheta)];

    % Weight linear and angular
    weight_linear = 100;
    weight_angular = 1;
    W = diag([weight_linear, weight_angular]);

    % Solve qp
    % qp_H = 2*eye(2);
    % qp_f = -2*u_desired;
    qp_H = 2*(W'*W);
    qp_f = -2*(W'*W)*u_desired;
    qp_A = -1 * Lgh;
    qp_b = -1 * -2*h_current;

    qp_options = optimoptions("quadprog", "Display", "off");
    u = quadprog(qp_H, qp_f, qp_A, qp_b, [], [], [], [], [], qp_options);
    u_valid = calc_acceptable_command(robotarium, u, 1.1);
    
    % u_valid = u_desired;

    if(~isequal(u, u_valid))
        fprintf("Slowed command from [%.2f,%.2f] to [%.2f,%.2f].\n", u(1), u(2), u_valid(1), u_valid(2));
        error("kys command constraint violated");
    end

    robotarium.set_velocities(1, u_valid);

    fprintf("h = %+.4E, u=[%7.4f,%7.4f]\n", h_current, u_valid(1), u_valid(2));

    % Check for waypoint advance
    if(norm(rpos - waypoint) <= waypoint_advance_distance)
        waypoint = [0;0] - waypoint;
        fprintf("Next waypoint: [%g, %g]\n", waypoint(1), waypoint(2));
        star_patch.XData = waypoint(1) + starX;
        star_patch.YData = waypoint(2) + starY;
        fprintf("Advancing waypoint!\n");
        sound(sound_laughter);
    end

    robotarium.step();
    if(real_time)
        pause(0.01);
    end
end


%% Helper Functions

% safety factor should be > 1
function acceptable_command = calc_acceptable_command(robotarium, command, safety_factor)
    % Code taken from robotarium
    r = robotarium.wheel_radius;
    l = robotarium.base_length;
    wheel_velocities = [(1/(2*r))*(2*command(1, :) - l*command(2, :)); ...
        (1/(2*r))*(2*command(1, :) + l*command(2, :))];
    max_wheel_velocity = max(abs(wheel_velocities));
    max_acceptable_wheel_velocity = robotarium.max_wheel_velocity / safety_factor;
    acceptable_command = command * min(1, max_acceptable_wheel_velocity/max_wheel_velocity);
end