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

max_iter = 100000;              % maximum number of iterations for gradient descent.
tol      = 1e-5;                % residula tolerance for gd.
tau      = 0.9/sqrt(8);         % step size for gd.
verbose  = 0;                   % print energy and residul.
volume   = 100*sum(mask(:));    % the volume of result surface.

%% Include mex file
try
    addpath(genpath('build'))
catch ME
    sprintf('Building c++ file is required!')
    return
end
%% Ballooning part
% initialization for ballooning
z_init = zeros(size(mask));

% compute boundary in this mask. Needed for projection to fulfill dirichlet
% assumption
boundary = logical(mask-imerode(mask,strel('disk', 1, 0)));

% ballooning part. Note that mex file assumes data to be row-major and
% matlab assumes data to be column major, thus the transposes.
ms = MinimalSurfaceMEX('initMinimalSurface', mask', single(z_init'), boundary', max_iter, tol, verbose);
z_ms = double(transpose(MinimalSurfaceMEX('GradientDescent', ms, single(z_init'), single(tau), scale_volume)));
MinimalSurfaceMEX('closeMinimalSurface', ms);

%% Visualization
figure (1);
imShow('depth3d', z_ms, mask, size(z_ms));
