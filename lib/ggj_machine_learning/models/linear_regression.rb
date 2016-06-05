
##
# h(x) = theta0 + theta1x
#
# Cost function: [square error function]
# J(theta0, theta1) = 1/2m [sum(i=0;m)((h(x)(i) - y(i))^2)]
#
# minimize (theta0, theta1) J(theta0, theta1)
#
# noinspection RubyInstanceMethodNamingConvention
# noinspection RubyInstanceVariableNamingConvention
class LinearRegression
  @m = 0
  @x, @y = []
  @theta0, @theta1 = 0

  def initialize(theta0 = 0, theta1 = 0)
    @theta0 = theta0
    @theta1 = theta1

    @m = 0
    @x = []
    @y = []

    initialize_descent_gradient
  end

  def h(x)
    @theta0 + @theta1 * x
  end

  def set_data(points)
    points.each do |point|
      @x << point[0]
      @y << point[1]
      @m += 1
    end

    @m
  end

  # Gradient Descent Method
  # Generalization of Cost Function
  # h(x) = theta0 + theta1X + theta2X^2 + ... thetaNX*N

  # J(theta0 ... thetaN) = 1/2m [sum(i=0;m)((h(x)(i) - y(i))^2)]
  # minimize (theta0 ... thetaN) = J(theta0 ... thetaN)

  # repeat until convergence
  # alpha = learning rate
  # thetaJ = thetaJ - alpha * deriv(J(theta0, theta1))

  # theta0 = theta0 - alpha * 1/m * [sum(i=1;m) (h(x(i)) - y(i))]
  # theta1 = theta1 - alpha * 1/m * [sum(i=1;m) (h(x(i)) - y(i)) * x(i)]
  ALPHA = 0.1
  PRECISION = 0.00001
  @precision
  @alpha
  def initialize_descent_gradient
    @precision = PRECISION
    @alpha = ALPHA
  end

  def gradient_descent_method(alpha = ALPHA, precision = PRECISION)
    @alpha = alpha
    @precision = precision

    old_theta0 = 9999999
    old_theta1 = 9999999
    until stop_gradient(old_theta0, old_theta1) do
      old_theta0 = @theta0
      old_theta1 = @theta1

      new_theta0_value = new_theta0
      new_theta1_value = new_theta1

      @theta0 = new_theta0_value
      @theta1 = new_theta1_value
    end
    [@theta0, @theta1]
  end

  private
  def new_theta0
    @theta0 - @alpha.to_f * sum_theta0 / @m
  end

  def new_theta1
    @theta1 - @alpha.to_f * sum_theta1 / @m
  end

  def sum_theta0
    sum = 0
    @m.times do |i|
      sum = sum + h(@x[i]) - @y[i]
    end

    sum
  end

  def sum_theta1
    sum = 0
    @m.times do |i|
      sum += (h(@x[i]) - @y[i]) * @x[i]
    end

    sum
  end

  def stop_gradient(old_theta0, old_theta1)
    (@theta0 - old_theta0).abs < @precision and
        (@theta1 - old_theta1).abs < @precision
  end
end