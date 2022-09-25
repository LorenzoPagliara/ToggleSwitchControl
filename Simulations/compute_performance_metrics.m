function [ISE,ITAE] = compute_performance_metrics(avg_LacI,avg_TetR, LacI_ref, TetR_ref)
    
    e1_bar = zeros(1, 1440);
    e2_bar = zeros(1, 1440);
    
    for i=1:length(avg_LacI)
        e1_bar(i) = ((avg_LacI(i) - LacI_ref)/LacI_ref);
        e2_bar(i) = ((avg_TetR(i) - TetR_ref)/TetR_ref);
    end
    
    e_bar = zeros(1, length(e1_bar));
    
    for i=1:length(e1_bar)
        e_bar(i) = norm([e1_bar(i), e2_bar(i)]);
    end
    ISE = sum(e_bar.^2);
    
    e_abs = zeros(1, length(e_bar));
    
    for i=1:length(e_bar)
        e_abs(i) = abs(e_bar(i))*i;
    end    
    ITAE = sum(e_abs);
   
end

