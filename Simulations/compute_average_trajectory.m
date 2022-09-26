function [avg_LacI, avg_TetR] = compute_average_trajectory(LacI, TetR, avg_time)

    avg_LacI = zeros(1, 1440/240);
    avg_TetR = zeros(1, 1440/240);
    
    
    
    for i = 1:240:1440
        j = fix((i - 1)/ 240) + 1;
        avg_LacI(j) = mean(LacI(i:i+240));
        avg_TetR(j) = mean(TetR(i:i+240));
    end
    
    x = (0:240:1440-1);    
    avg_LacI = spline(x,avg_LacI,avg_time);
    avg_TetR = spline(x,avg_TetR,avg_time);

end