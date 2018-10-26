function Ground_truth_export()
%   Detailed explanation goes here
    Groundtruth = csvread('/home/nigno/Robots/pytorch_tests/MirkoNet/dataRegression3/test_ot_nino.csv');
    mark_pos = Groundtruth(:,2:4);
    gl_quat = Groundtruth(:,5:8);
    gl_pos = Groundtruth(:,9:11);
    ts = Groundtruth(:,1);
    num_el = size(gl_quat);
    num_el(1)
    for i = 1:num_el(1)
        gl_rotm(:,:,i) = qGetR(gl_quat(i,:));
    end
    
    for i =  1 : size(Groundtruth,1)
        gl_rotmi = gl_rotm(:,:,i);
        gl_rel_pos(i,:) =  (gl_rotmi'*(mark_pos(i,:) - gl_pos(i,:))')';
    end
    Groundtruth_rel(:,1) = ts;
    Groundtruth_rel(:,2:4) = gl_rel_pos;
    
    plot (Groundtruth_rel(:,2:4))
    
    csvwrite('/home/nigno/Robots/pytorch_tests/MirkoNet/dataRegression3/test_ot_nino_rel.csv',Groundtruth_rel);
end