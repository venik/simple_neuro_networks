#!/usr/bin/octave-cli -qf

# Sasha Nikiforov, k-means naive implementation
# https://en.wikipedia.org/wiki/K-means_clustering

fprintf("y0y0\n");

samples = 1000;
sigma = 2;
x = zeros(samples, 3);
x(1:samples / 2, :) = sigma * randn(samples / 2 , 3) +  [1.5, 1.5, 1.5];
x(samples / 2 + 1 : samples, :) = sigma * randn(samples / 2 , 3) +  [6.75, 2.5, 6.75];


S1 = zeros(length(x), 1);
S2 = zeros(length(x), 1);

iterations = 10;
centroids = zeros((iterations + 1) * 2, 3);

# it's not so good idea to pick up first two samples, they should be random
# but it's POC code
m1 = centroids(1, :) = x(1, :);
m2 = centroids(2, :) = x(2, :);

centroid_num = 3;
for iter=1:iterations
    num_in_s1 = 0;
    num_in_s2 = 0;

    # Assignment step
    for k=1:length(x)
        sample = x(k, :);
        norm_to_s1 = norm(sample - m1, 2);
        norm_to_s2 = norm(sample - m2, 2);
        if (norm_to_s1 > norm_to_s2)
            # assign to S2
            num_in_s2 = num_in_s2 + 1;
            S2(num_in_s2) = k;
        else
            num_in_s1 = num_in_s1 + 1;
            S1(num_in_s1) = k;
        end
    end
    
    # Update centroids step
    for k_s1=1:num_in_s1
        centroids(centroid_num, :) = centroids(centroid_num, :) + x(S1(k_s1), :);
    end
    centroids(centroid_num, :) = centroids(centroid_num, :) / num_in_s1;

    for k_s2=1:num_in_s2
        centroids(centroid_num + 1, :) = centroids(centroid_num + 1, :) + x(S2(k_s2), :);
    end
    centroids(centroid_num + 1, :) = centroids(centroid_num + 1, :) / num_in_s2;

    if (centroids(centroid_num, :) == centroids(centroid_num - 2, :)) and (centroids(centroid_num + 1, :) == centroids(centroid_num - 1, :));
        break;
    end

    # update centroids
    m1 = centroids(centroid_num, :);
    m2 = centroids(centroid_num + 1, :);

    centroid_num = centroid_num + 2;
end

iter

centroids(1:2:centroid_num-1,:)
centroids(2:2:centroid_num-1,:)

plot3(x(:, 1), x(:, 2), x(:,3), 'o',
        centroids(1:2:centroid_num-1, 1), centroids(1:2:centroid_num-1, 2), centroids(1:2:centroid_num-1, 3),
            'gx', 'markersize', 15, 'LineWidth', 3,
        centroids(2:2:centroid_num-1, 1), centroids(2:2:centroid_num-1, 2), centroids(2:2:centroid_num-1, 3),
            'mx', 'markersize', 15, 'LineWidth', 3,
        centroids(centroid_num:centroid_num+1, 1), centroids(centroid_num:centroid_num+1, 2), centroids(centroid_num:centroid_num+1, 3),
            'r*', 'markersize', 15, 'LineWidth', 3);
ylim([0 5]), xlim([0 10]), zlim([0 10]);
grid on;
pause();

fprintf("end\n")
