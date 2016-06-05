require "ggj_machine_learning/version"
require "ggj_machine_learning/models/linear_regression"

class GgjMachineLearning
  @model = nil
  @points = []

  def initialize
    @points = []
  end

  def self.helloWorld
    "Hello World!"
  end

  def setModel
    @model = LinearRegression.new
  end

  def learn(points)
    @points += points

    @model.set_data(@points)

    @model.gradient_descent_method
  end

  def predict(x)
    @model.h(x)
  end
end
