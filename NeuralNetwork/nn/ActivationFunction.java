package nn;

//basic hard or soft threshold function like linear and logistic Classifier
//use to normalize the activation of each neuron value
public interface ActivationFunction {
	double activation(double z);

	double derivative(double z);
}
