require 'spec_helper'

describe GgjMachineLearning do
  it 'has a version number' do
    expect(GgjMachineLearning::VERSION).not_to be nil
  end

  it 'Hello World' do
    expect(GgjMachineLearning.helloWorld).to eq "Hello World!"
  end
end
