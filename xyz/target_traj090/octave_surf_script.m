
% Clear all variables and close any existing figures
clear;
close all;

% load data and make PCD
A = load('combined_iam_file.dat','-ascii');
I0 = A(:, 1);
pcd = 100 * (A - I0) ./ I0;
q = linspace(0, 8, 81);
t = linspace(0, 75*2.6, 76);
disp(['max(PCD) = ', num2str(max(max(pcd)))]);
disp(['min(PCD) = ', num2str(min(min(pcd)))]);

% Create the surf plot
figure;
surf(t, q, pcd(1:81, :));

% Customize the plot (optional)
%title('Surf Plot Example');
xlabel('t (fs)');
ylabel('q (Å^{-1})');
zlabel('%ΔI(q,t)');
caxis([-35 60]);
zlim([-35 60]);
%colormap hot
colormap default
colorbar("location", "NorthOutside");  % Add a colorbar
%shading interp;  % Interpolate shading

% Set view angle (optional)
view(-35, 60);  % Adjust the view angle if needed

% Specify the DPI for the output PNG image
dpi = 300;  % Change this value to your desired DPI

% Save the plot as a PNG image with the specified DPI
print('surf_plot.png', '-dpng', sprintf('-r%d', dpi));

% Optional: Display a message to indicate completion
disp(['Surf plot saved as surf_plot.png with DPI = ', num2str(dpi)]);
