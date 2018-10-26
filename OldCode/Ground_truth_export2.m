function Ground_truth_export2()
    format long

%   Detailed explanation goes here
    %GroundtruthOld = csvread('/home/nigno/Robots/pytorch_tests/MirkoNet/dataRegression4/test_ot_nino.csv');
    GroundtruthOld = csvread('/media/nigno/Data/mirko dataset/008/S9P8XR~E.CSV');
    new_size = floor(size(GroundtruthOld, 1) / 2.0)
    
    for i = 1:new_size
        Groundtruth(i,:) = GroundtruthOld(i,:);
        j = ((i - 1) * 2) + 1;
        Groundtruth(i,12:13) = GroundtruthOld(j,12:13);
    end
    
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
 
    dlmwrite('/media/nigno/Data/mirko dataset/008/test_ot_nino2.csv', Groundtruth, 'precision', '%10.5f');
    csvwrite('/media/nigno/Data/mirko dataset/008/test_ot_nino_rel.csv',Groundtruth_rel);
    %dlmwrite('/home/nigno/Robots/pytorch_tests/MirkoNet/dataRegression4/test_ot_nino2.csv', Groundtruth, 'precision', '%10.5f');
    %csvwrite('/home/nigno/Robots/pytorch_tests/MirkoNet/dataRegression4/test_ot_nino_rel.csv',Groundtruth_rel);
end