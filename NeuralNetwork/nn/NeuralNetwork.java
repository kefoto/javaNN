package nn;

import java.util.List;

import java.util.ArrayList;
import java.util.Collections;

/*
 * @author: Ke Xu
 * Kxu27@u.rochester.edu
 */

public class NeuralNetwork {
	
	private ActivationFunction af = new LogisticAF();
	
	private int generationVersion;
	
	private Layer[] hiddenLayers;
	
	//the value of each neuron
	private double[][] activations;
	
	//editable
	//training sample size = 2 * this sample size
	private int sampleSize = 2500;
	
	//for graph illustration
	public ArrayList<String> lines = new ArrayList<>();
	
	
	public NeuralNetwork(int... dimensions) {
		activations = new double[dimensions.length][];
		
		//one size smaller than activation matrix
		hiddenLayers = new Layer[dimensions.length - 1];
		
		activations[0] = new double[dimensions[0]];
		
		for (int i = 1; i < dimensions.length; i++) {
            activations[i] = new double[dimensions[i]];
            hiddenLayers[i - 1] = new Layer(dimensions[i], dimensions[i - 1]);
        }
	}
	
	public NeuralNetwork(NeuralNetwork other) {
        generationVersion = other.generationVersion;
//        fileName = other.fileName;

        double[][] otherActivations = other.activations;
        activations = new double[otherActivations.length][];
        activations[0] = new double[otherActivations[0].length];

        Layer[] otherHiddenLayers = other.hiddenLayers;
        hiddenLayers = new Layer[activations.length - 1];

        for (int i = 1; i < activations.length; i++) {
            activations[i] = new double[otherActivations[i].length];
            hiddenLayers[i - 1] = new Layer(otherHiddenLayers[i - 1]);
        }
    }
	
	
	/* Propagate the inputs forward to compute the outputs */
	//calculate the whole network, with a initiated randomized network
	public void process(double[] inputs) {
		if (inputs.length != activations[0].length) {
            throw new IllegalArgumentException("Invalid number of inputs " + inputs.length);
        }
		
		System.arraycopy(inputs, 0, activations[0], 0, inputs.length);
		
		for(int i = 0; i < activations.length - 1; i++) {
			activations[i + 1] = hiddenLayers[i].fCalculate(activations[i]);
			
		}	
	}
	/*
	 * Parameters:
	 * 		Training examples set,
	 * 		epochs: number of repeats on analyzing randomized training set,
	 * 		batch size,
	 * 		alpha training rate (from linear classifier)
	 */
	public void train(List<Example> examples, int steps, int trainingSize, LearningRateSchedule schedule){
		
		//train by random samples with sample size
		for(int i = 0; i < steps; i++) {		
			Collections.shuffle(examples);
			List<Example> selection = examples.subList(0, trainingSize);
			for(Example e: selection) {

				//the weights of each layer are already randomized through object construction
				//forward prop
				process(e.inputs);	
						
				//backward prop
				//output from one example
				backPropagation(e.output, schedule);	
	
			}	
		}		
	}
	
	//update the delta and then every weight
	public void backPropagation(double[] output,  LearningRateSchedule schedule) {
		double[] outputError = new double[output.length];
        double[] outputDelta = new double[output.length];

		for(int i = 0; i < output.length; i++) {
			//last layer of the activation layer is the predicted amount for the output
			outputError[i] = output[i] - activations[activations.length - 1][i];
			
			outputDelta[i] = outputError[i] * af.derivative(activations[activations.length - 1][i]);
		}
		
		for(int i = hiddenLayers.length - 1; i >= 0; i--) {
			//????
			if (i == hiddenLayers.length - 1) {
				hiddenLayers[i].updateDelta(outputDelta, activations[activations.length - 1]);				
			} else {
				hiddenLayers[i].updateDelta(hiddenLayers[i + 1].getDeltas(), activations[i + 1]);			
			}	
			
//			for(double x: hiddenLayers[i].getDeltas()) {
//				System.out.print(x + " ");
//			}
//			System.out.println();
		}
		for(int i = 0; i < hiddenLayers.length - 1; i++) {
			if (i == hiddenLayers.length - 1) {
				hiddenLayers[i].updateWeight(activations[i], outputDelta, 1);
			} else {
				hiddenLayers[i].updateWeight(activations[i], hiddenLayers[i + 1].getDeltas(), schedule.alpha(i));
			}
		}
	}
	
	public int error(List<Example> examples) {
		int nerror = 0;
		for (Example ex : examples) {
			process(ex.inputs);
			int index = getActivatedOutput();
			int actualClass = -1;
			for(int i = 0; i < ex.output.length; i++) {
				if(ex.output[i] == 1.0) {
					actualClass = i;
				}
			}
			
			if (index != actualClass) {
				nerror += 1;
			}
		}
		return nerror;
	}
	public double accuracy(List<Example> examples) {
		int ncorrect = 0;
		for (Example ex : examples) {
			process(ex.inputs);
			int index = getActivatedOutput();
			int actualClass = -1;
			for(int i = 0; i < ex.output.length; i++) {
				if(ex.output[i] == 1.0) {
					actualClass = i;
				}
			}
			
			if (index == actualClass) {
				ncorrect += 1;
			}
		}
		return (double)ncorrect / examples.size();
	}
	
	//built only for mnist graph illustration
	//takes additional testing set that build the outputs in this class instead of the main class
	public void train2(List<Example> examples, List<Example> testing, int steps, LearningRateSchedule schedule){
		 
		for(int i = 0; i < steps; i++) {
			Collections.shuffle(examples);
			List<Example> selection = examples.subList(0, sampleSize * 2);
			for(Example e: selection) {

				//the weights of each layer are already randomized through object construction
				//forward prop
				process(e.inputs);	
						
				//backward prop
				//output from one example
				backPropagation(e.output, schedule);	
	
			}
			if(i % 2 == 0) {
				Collections.shuffle(testing);
				List<Example> selectiont = testing.subList(0, sampleSize);
				String line = Integer.toString(i) + "," + Integer.toString(error(selectiont));
				 lines.add(line);
			}
		System.out.print(i + " ");
		}		
	}
	//getters and setters
	
	public double[] getOutputs() {
        return activations[activations.length - 1];
    }
	
	public int getActivatedOutput() {
        double[] outputs = getOutputs();
        int result = 0;
        double maxOutput = outputs[0];

        for (int i = 1; i < outputs.length; i++) {
            if (outputs[i] > maxOutput) {
                maxOutput = outputs[i];
                result = i;
            }
        }

        return maxOutput == 0 ? -1 : result;
    }
	
	public Layer[] getHiddenLayers() {
        return hiddenLayers;
    }

    public int getGenerationVersion() {
        return generationVersion;
    }

    public void incrementGV() {
        generationVersion++;
    }

    public double[][] getActivations() {
        return activations;
    }
}
