require 'spec_helper'

describe "Linear Regression" do

  it 'H - Empty' do
    linear_regression = LinearRegression.new
    y = linear_regression.h(99999)

    expect(y).to be 0
  end

  it 'H - Right Theta 0' do
    linear_regression = LinearRegression.new (10)
    y = linear_regression.h(99999)

    expect(y).to be 10
  end

  it 'H - Right Theta 1' do
    linear_regression = LinearRegression.new(0, 2)
    y = linear_regression.h(5)

    expect(y).to be 10
  end

  it 'H - Right Theta 1' do
    linear_regression = LinearRegression.new(1, 2)
    y = linear_regression.h(5)

    expect(y).to be 11
  end

  it 'Set Data' do
    linear_regression = LinearRegression.new(1,1)
    points = [
        [1,2]
    ]
    m = linear_regression.set_data(points)

    expect(m).to be points.length
  end

  it 'Sum Theta0' do
    linear_regression = LinearRegression.new
    linear_regression.stub(:h) do
      1
    end

    y = [1, 2, 3]
    linear_regression.instance_variable_set(:@x, y)
    linear_regression.instance_variable_set(:@y, y)
    linear_regression.instance_variable_set(:@m, y.length)


    tetha1 = linear_regression.send(:sum_theta0)

    expect(tetha1).to be -3
  end

  it 'Sum Theta1' do
    linear_regression = LinearRegression.new
    linear_regression.stub(:h) do
      1
    end

    y = [1, 2, 3]
    linear_regression.instance_variable_set(:@x, y)
    linear_regression.instance_variable_set(:@y, y)
    linear_regression.instance_variable_set(:@m, y.length)


    tetha1 = linear_regression.send(:sum_theta1)

    expect(tetha1).to be -8
  end

  it 'Stop Gradient - True' do
    linear_regression = LinearRegression.new(1,1)
    theta0 = 1
    theta1 = 1

    stop = linear_regression.send(:stop_gradient, theta0, theta1)

    expect(stop).to be_truthy
  end

  it 'Stop Gradient - False (tetha0)' do
    linear_regression = LinearRegression.new(1,1)
    theta0 = 3
    theta1 = 1.5

    stop = linear_regression.send(:stop_gradient, theta0, theta1)

    expect(stop).to be_falsey
  end

  it 'Stop Gradient - False (tetha1)' do
    linear_regression = LinearRegression.new(1,1)
    theta0 = 1.5
    theta1 = 3

    stop = linear_regression.send(:stop_gradient, theta0, theta1)

    expect(stop).to be_falsey
  end

  it 'Stop Gradient - False (Both)' do
    linear_regression = LinearRegression.new(1, 1)
    theta0 = 3
    theta1 = 3

    stop = linear_regression.send(:stop_gradient, theta0, theta1)

    expect(stop).to be_falsey
  end

  it 'Gradient Descent Method' do
    linear_regression = LinearRegression.new

    points = [
        [1,1],
        [2,2]
    ]
    linear_regression.set_data(points)

    theta = linear_regression.gradient_descent_method


    expect(theta[0]).to be_between(-0.005, 0.005)
    expect(theta[1]).to be_between(0.995, 1.005)
  end
end