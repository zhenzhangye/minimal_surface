clear all;
%% Data prepartion
dataset = 'data/xtion_backpack_sf4_ups.mat';
  
try
    load(dataset);
catch ME
    sprintf('Dataset is not found! Run the script in data folder')
    return
end

mask = mask_sr;
clear albedo_est I_noise K_lr K_sr mask_lr mask_sr z0_noise z_est;

options.max_iter = 100000;              % maximum number of iterations for gradient descent.
options.tol      = 1e-5;                % residula tolerance for gd.
options.tau      = 0.9/sqrt(8);         % step size for gd.
options.verbose  = 0;                   % print energy and residul.
options.scale_volume   = 100;    % the volume of result surface.

%% Include mex file
try
    addpath(genpath('build'))
catch ME
    sprintf('Building c++ file is required!')
    return
end
%% Ballooning part
z_ms = minimalSurface(mask,options);

%% Visualization
figure (1);
imShow('depth3d', z_ms, mask, size(z_ms));
