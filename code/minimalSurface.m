function z_ms = minimalSurface(mask,options)
%% Ballooning part
% initialization for ballooning
z_init = zeros(size(mask));

% compute boundary in this mask. Needed for projection to fulfill dirichlet
% assumption
boundary = logical(mask-imerode(mask,strel('disk', 1, 0)));

volume_sphere = sum(mask(:));

% ballooning part. Note that mex file assumes data to be row-major and
% matlab assumes data to be column major, thus the transposes.
ms = MinimalSurfaceMEX('initMinimalSurface', mask', single(z_init'), boundary', options.max_iter, options.tol, options.verbose);
z_ms = double(transpose(MinimalSurfaceMEX('GradientDescent', ms, single(z_init'), single(options.tau), volume_sphere*options.scale_volume)));
MinimalSurfaceMEX('closeMinimalSurface', ms);

end

