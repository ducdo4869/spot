function mu_new = update_mu(mu_prev, mu_prev2, X, weights, y, y_h, sigma, eta, alpha)
    % mu_prev: ma trận tâm tại bước k-1 (h x n)
    % mu_prev2: ma trận tâm tại bước k-2 (h x n)
    % X: vector đầu vào (1 x n)
    % weights: vector trọng số (h x 1)
    % y: giá trị đầu ra mô phỏng
    % y_h: đầu ra mạng RBF
    % sigma: vector độ rộng (h x 1)
    % eta: hệ số học
    % alpha: hệ số quán tính
    % mu_new: ma trận tâm cập nhật (h x n)

    [h, n] = size(mu_prev);
    mu_new = zeros(h, n);

    for j = 1:h
        for i = 1:n
            delta_mu = eta * (y - y_h) * weights(j) * (X(i) - mu_prev(j,i)) / (sigma(j)^2);
            mu_new(j,i) = mu_prev(j,i) + delta_mu + alpha * (mu_prev(j,i) - mu_prev2(j,i));
        end
    end
end
