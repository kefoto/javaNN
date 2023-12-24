package nn;

public class LogisticAF implements ActivationFunction{
	public double activation(double z) {
		return 1.0/(1.0 + Math.exp(-z));
	}
	
	//returns the derivative of this activation function, which is the sigmoid function
	public double derivative(double z) {
		double e = activation(z);
		return e * (1.0 - e);
	}
}
