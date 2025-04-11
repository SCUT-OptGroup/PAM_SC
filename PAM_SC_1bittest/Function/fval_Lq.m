%% This file is calculating thr Lq norm, where q \in [0,1]

function fval = fval_Lq(g,q)

if q == 1 || q == 0 || q==1/2 || q == 2/3
    switch q
        case 0
           fval =  sum(g>1.0e-8);
        case 1
           fval =  sum(g);
        case 0.5
           fval =  sum(g.^(1/2));
        case 2/3
           fval =  sum(g.^(2/3));
    end
    
end