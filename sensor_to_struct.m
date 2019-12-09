% Loads sensor data from MATLAB Mobile and converts it to scipy interpretable format %
% Usage ex: convert 'MobileSensorData/d1.mat' 'test.mat' %
function sensor_to_struct(readfile, writefile)
    acceleration = load(readfile).Acceleration;
    
    % Convert time to seconds
    d_t = posixtime(acceleration.Timestamp);
    d_t = d_t - d_t(1);
    
    % Build struct
    s = struct('time', d_t, 'x', acceleration.X, 'y', acceleration.Y, ...
               'z', acceleration.Z);
    
    % Write to disk
    save(writefile, 's');
end

