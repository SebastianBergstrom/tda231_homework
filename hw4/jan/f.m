function ret = f(alpha, t, x)
  N = length(alpha);
  ret = sum(alpha);
  for i = 1:N
    for j = 1:N
      ret = ret - alpha(i)*alpha(j)*t(i)*t(j)*x(i,:)*x(j,:)';
    end
  end
end
