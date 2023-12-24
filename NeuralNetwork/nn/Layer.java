package nn;

import java.util.Random;

/*
 * @author: Ke Xu
 * Kxu27@u.rochester.edu
 */

public class Layer {
	
	private ActivationFunction af = new LogisticAF();
	
	//weight matrix between the previous size layer and current size layer
	private double[][] weights;
	private double[] deltas;
	//private double[] biases;
	
	public Layer(int currentSize, int previousSize) {
        weights = new double[currentSize][previousSize];
        deltas = new double[previousSize];
        
        double variance = 2.0 / (currentSize + previousSize);
        Random random = new Random();
        
        //randomize the weights and biases
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < weights[0].length; j++) {
                weights[i][j] =random.nextGaussian() * Math.sqrt(variance); // [-1 .. 1)
            }
        }
    }
	
	public Layer(Layer other) {
		double[][] tempWeights = other.weights;
        double[] tempDelta = other.deltas;
        weights = new double[tempWeights.length][tempWeights[0].length];
        deltas = new double[tempDelta.length];

        deltas = tempDelta.clone();
        
        for (int i = 0; i < weights.length; i++) {
            System.arraycopy(tempWeights[i], 0, weights[i], 0, weights[0].length);
        }
		
	}
	
	//getting the next layer's activation based on weight and biases
	//normalized
	//forward propagation
	public double[] fCalculate(double[] activation) {
		//previous size
		 if (activation.length != weights[0].length) {
	            throw new IllegalArgumentException("Invalid activation length " + activation.length);
	     }
		 
		 double[] result = new double[weights.length];
		 
		 for (int i = 0; i < weights.length; i++) {
			 double sum = 0;
			 
	         for (int j = 0; j < weights[0].length; j++) {
	        	 sum += activation[j] * weights[i][j];
	         }
	         
	         result[i] = af.activation(sum);
	     }

	        return result;
	}
	
	//for backward propagation
	//Argument: current layer delta, and current layer activation
	//update the delta between previous layer and updates the weights
	public void updateDelta(double[] delta, double[] activation) {
		
		//if this is the same as the current size
		if (delta.length != weights.length) {
            throw new IllegalArgumentException("Invalid activation length " + delta.length);
		}
		if (activation.length != weights.length) {
            throw new IllegalArgumentException("Invalid activation length " + activation.length);
		}	
		
		for (int i = 0; i < weights[0].length; i++) {
			double sum = 0;
			double sum2 = 0;
			
			 
	         for (int j = 0; j < weights.length; j++) {
	        	 sum += weights[j][i] * delta[j];
	        	 sum2 += weights[j][i] * activation[j];
	         }
	         deltas[i] = af.derivative(sum2) * sum;
		}
		
	}
	
	public void updateWeight(double[] activation, double[] delta, double alpha) {
		 for (int i = 0; i < weights.length; i++) {
	         for (int j = 0; j < weights[0].length; j++) {
	        	 weights[i][j] += alpha * activation[j] * delta[i];
	         }
	     }
	}
	
	//getters and setters
	public void setDeltas(double[] deltas) {
        this.deltas = deltas;
    }
	
	public double[] getDeltas() {
		return deltas;
	}
	
	public String toString() {
		String result = "";
		for(int i = 0; i < weights.length; i++) {
			for(int j = 0; j < weights[0].length; j++) {
				result += "[" + Integer.toString(i) + Integer.toString(j) + ": " + Double.toString(weights[i][j]) + "] \t";
			}
			result += "\n";
		}
		
		return result;
	}
}
