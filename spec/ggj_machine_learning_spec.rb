require 'spec_helper'

describe GgjMachineLearning do
  it 'has a version number' do
    expect(GgjMachineLearning::VERSION).not_to be nil
  end

  it 'Hello World' do
    expect(GgjMachineLearning.helloWorld).to eq "Hello World!"
  end

  it 'Set Model' do
    m_learning = GgjMachineLearning.new

    m_learning.setModel

    model = m_learning.instance_variable_get(:@model)

    expect(model.class).to be(LinearRegression)
  end

  it 'Learn - Add more points' do
    m_learning = GgjMachineLearning.new
    m_learning.setModel

    m_learning.instance_variable_get(:@model).stub(:gradient_descent_method) do
      1
    end

    points1 = [
        [0,0],
        [1,1]
    ]
    m_learning.learn(points1)

    points2 = [
        [2,2],
        [3,3]
    ]
    m_learning.learn(points2)

    m_points = m_learning.instance_variable_get(:@points)

    expect(m_points.length).to be 4
  end
end
